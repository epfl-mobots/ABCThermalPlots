'''
Author(s):
    Cyril Monette, EPFL, cyril.monette@epfl.ch
'''

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from ABCImaging.VideoManagment.videolib import generateVideoFromList, fig_to_rgb_array, cropFrameToContent
from ABCImaging.HiveOpenings.libOpenings import valid_ts

def generateThermalDF(df:pd.DataFrame)->pd.DataFrame:
    '''
    Generates a panda df for temperatures that is friendly with the thermalutil.py libray.
    This means the columns are t00-t64 and one line is one timestamp.
    Parameters:
    - df: pd.DataFrame containing the temperatures in an influxdb format.

    returns:
    - upper: pd.DataFrame containing the upper hive temperatures
    - lower: pd.DataFrame containing the lower hive temperatures
    '''
    _index = df.index.unique()
    upper = pd.DataFrame(index=_index)
    lower = pd.DataFrame(index=_index)
    df_upper = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'upper')]
    df_lower = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'lower')]

    for i in range(64):
        column_name = f't{i:02d}'
        # For every index of thermal_df, get the temperature which has the same datetime in df
        upper[column_name] = df_upper[df_upper['_field'] == column_name]['_value']
        lower[column_name] = df_lower[df_lower['_field'] == column_name]['_value']

    # Set the right column names
    upper.columns = [f't{i:02d}' for i in range(64)]
    lower.columns = [f't{i:02d}' for i in range(64)]

    # Convert every column to float type
    for col in upper.columns:
        upper[col] = upper[col].astype(float)
    for col in lower.columns:
        lower[col] = lower[col].astype(float)

    # Suppress eratic values
    upper[upper < -50] = np.nan
    lower[lower < -50] = np.nan
    return upper, lower

def readFromFile(filepath:str, verbose:bool=False)->pd.DataFrame:
    '''
    Reads a .dat or .csv file and returns a pandas dataframe with the thermal data.
    The file is expected to have the following format:
    - First column: datetime in ISO format
    - Next 64 columns: temperature data from sensors t00 to t63
    - Optional next column: validity flag (True/False)
    - Optional next 10 columns: target temperatures
    - Optional next 10 columns: h_avg_temps
    - Optional next 10 columns: pwm values
    
    Parameters:
    - filepath: path to the .dat file
    
    Returns:
    - df_therm_data: pandas dataframe containing thermal data (datetime as index, 64 columns for sensor data and potentially one more column for validity)
    '''

    assert filepath.endswith('.dat') or filepath.endswith('.csv'), "File must be a .dat or .csv file"

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

    else:
        # First skip the lines that do not start with "20" (the millenia)
        skip = 0
        with open(filepath, 'r') as f:
            lines = f.readlines()
            while skip < len(lines) and not lines[skip].startswith('20'):
                skip += 1

        try:
            # Assuming data is in a structured format like CSV or similar
            df = pd.read_csv(filepath, delimiter=',', skiprows=skip, header=None)
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")

    validity_flag = False
    # make the columns as being : datetime, 64 columns for sensor data
    temps_labels = [f't{i:02d}' for i in range(64)]
    if df.shape[1] == 1+ 64 + 1 + 10 + 10 + 10 or df.shape[1] == 1 + 64 + 1: # datetime + 64 temps + 1 validity + 10 targets + 10 h_avg_temps + 10 pwm
        validity_flag = True
        if verbose:
            print("File has validity flag")
        temps_labels.append('validity')
    other_cols = df.shape[1] - (1 + 64 + (1 if validity_flag else 0))
    other_cols_labels = []
    if other_cols > 0:
        other_cols_labels = ['target' for _ in range(10)] + ['h_avg_temps' for _ in range(10)] + ['pwm' for _ in range(10)]
    df.columns = ['datetime'] + temps_labels + other_cols_labels
    df.set_index('datetime', inplace=True)
    if verbose:
        print(f"dataframe preview:\n {df.head()}")

    temps = df.iloc[:,0:64]
    if validity_flag:
        #typecast the validity column to boolean
        df.loc[:,'validity'] = df.loc[:,'validity'].astype(bool)
        temps = temps[df.loc[:,'validity'] == True]
    # Delete the validity column if it exists
    if 'validity' in df.columns:
        df.drop(columns=['validity'], inplace=True)
    if verbose:
        print(f"temps preview:\n {temps.head()}")

    if df.shape[1] > 64: # 64 temps
        targets = df.iloc[:,64:74]
        targets.columns = [f'h{i:02d}' for i in range(targets.shape[1])]
    else:
        targets = pd.DataFrame(index=temps.index) # Empty dataframe
    if verbose:
        print(f"targets preview:\n {targets.head()}")

    return temps, targets

def preview(df_therm_data, 
            show_sensors:bool=False, 
            contours:list[float] = None, 
            vmin=None,
            vmax=None,
            rows=None):
    '''
    Preview the thermal data in the file through 6 plots of equally spaced time points.
    param df_therm_data: pandas dataframe containing thermal data (datetime as index, 64 columns for sensor data and potentially one more column for validity)
    param rows: list of row indices to preview
    '''
    # check that the df is valid
    if not isinstance(df_therm_data, pd.DataFrame):
        raise ValueError("The input is not a pandas dataframe")
    if df_therm_data.shape[1] != 65 and df_therm_data.shape[1] != 64:
        raise ValueError(f"The input dataframe does not have the correct shape (needs to be 64 or 65 columns and got {df_therm_data.shape[1]})")
    if df_therm_data.index.name != 'datetime':
        raise ValueError("The input dataframe does not have the correct index name (needs to be 'datetime')")
    if df_therm_data.iloc[:,0].name !='t00':
        raise ValueError("The input dataframe does not have the correct column names (first column should be 't00')")
    
    # iterate through a few rows
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 7))
    n_data_points = df_therm_data.shape[0]
    if rows is None:
        rows = np.linspace(0, n_data_points-1, 6, dtype=int)

    print("=====================================================================================================================================================")
    print(f"Previewing thermal data:")
    for i, row in enumerate(rows):
        temps = df_therm_data.iloc[row]
        temps_np = temps.to_numpy()
        try:
            tf = ThermalFrame(temperature_data=temps_np)

        except NoValidSensors as e:
            print(f"Skipping row {row} due to no valid sensors")
            continue
        
        tf.calculate_thermal_field(verbose=False)

        x, y = tf.get_max_temp_pos()

        print(f"Max temp at {x}, {y} for row {row} [tmax = {tf.max_temp} °C]")
        a = ax.flat[i]
        tf.plot_thermal_field(a,show_cb=True, show_sensors=show_sensors, show_max_temp=True, contours=contours, annotate_contours=True, v_min=vmin,v_max=vmax)
        a.set_title(f"Row {row} -> {str(df_therm_data.iloc[row].name)}") # Overwrite the title

def generateThermalVideo(df_therm_data:pd.DataFrame, 
                         video_name:str, 
                         fps:int=10,
                         hive_nb:int=-1,
                         show_cb:bool=False,
                         show_sensors:bool=False, 
                         show_max_temp:bool=False,
                         contours:list[float]=[], 
                         vmin:float=None, vmax:float=None,
                         faulty_s:list[int]=[],
                         padding:int=50,
                         verbose:bool=False):
    '''
    Generates a video from the thermal data in the dataframe.
    The smallest gap between two timestamps is used to determine the expected time resolution.
    '''

    time_diffs = df_therm_data.index.to_series().diff().dropna()
    time_res = time_diffs.min()
    if verbose:
        print(f"Time resolution of the data: {time_res}")
    time_index = pd.date_range(start=df_therm_data.index.min(), end=df_therm_data.index.max(), freq=time_res)

    # Put NaN values for faulty sensors
    for s in faulty_s:
        col_name = f't{s:02d}'
        if col_name in df_therm_data.columns:
            df_therm_data.loc[:,col_name] = np.nan
        else:
            print(f"Warning: sensor {s} is not in the dataframe columns")
    
    # Set the vmin and vmax if not provided
    if vmin is None:
        vmin = df_therm_data.min().min()
    if vmax is None:
        vmax = df_therm_data.max().max()

    if contours == []:
        contours = list(np.arange(np.floor(vmin), np.ceil(vmax), 1))

    # Find typical frame size:
    fig, ax = plt.subplots(figsize=(13, 7))
    _tf = ThermalFrame(temperature_data=df_therm_data.iloc[0].to_numpy(), hive_id=hive_nb)
    _tf.plot_thermal_field(ax, show_cb=show_cb, show_sensors=show_sensors, contours=contours, v_min=vmin, v_max=vmax)
    example_frame = fig_to_rgb_array(fig)
    example_frame = cropFrameToContent(example_frame, padding=padding)
    height, width, channels = example_frame.shape
    if verbose:
        print(f"Video frame size: {height}x{width}x{channels}")
    plt.close(fig) # We close the figure to avoid displaying it

    # Generate the frames:
    frames = []
    for t in tqdm(time_index, desc="Generating frames…"):
        if t not in df_therm_data.index:
            if verbose:
                print(f"Time {t} is not in the dataframe index")
            # Append black frame
            frames.append(np.zeros((height, width, channels), dtype=np.uint8))
            continue
        try:
            tf = ThermalFrame(temperature_data=df_therm_data.loc[t].to_numpy(), hive_id=hive_nb, ts=t, bad_sensors=faulty_s)
        except NoValidSensors as e:
            print(f"Skipping time {t} due to no valid sensors")
            # Append black frame
            frames.append(np.zeros((height, width, channels), dtype=np.uint8))
            continue
        tf.calculate_thermal_field(verbose=False)
        fig, ax = plt.subplots(figsize=(13, 7))
        tf.plot_thermal_field(ax, show_cb=show_cb, show_sensors=show_sensors, show_max_temp=show_max_temp, contours=contours, v_min=vmin, v_max=vmax)
        ax.set_title(f"Thermal field for hive {tf.hive_id} at {str(t)}") # Overwrite the title
        _frame = fig_to_rgb_array(fig)
        _frame = cropFrameToContent(_frame, padding=padding)
        frames.append(_frame)
        plt.close(fig) # We close the figure to avoid displaying it
    if verbose:
        print(f"Generated {len(frames)} frames for the video")

    # Generate the video
    generateVideoFromList(frames, dest='outputVideos', name=video_name, fps=fps, grayscale=False)


def trend_filter(data, lmbd=50, order=2):
    ''' Trend filtering
        type: 'L1' or 'HP'

        Refs:
        - https://towardsdatascience.com/introduction-to-trend-filtering-with-applications-in-python-d69a58d23b2
    '''
    import cvxpy as cp
    _data = None
    if isinstance(data, list):
        _data = np.array(data)
    elif isinstance(data, np.ndarray):
        _data = data
    else:
        print("Data seems to be in the wrong format!!!")

    # Regularization parameter
    # Convergence to original data as λ→0
    # Finite convergence to best affine fit as λ→∞
    vlambda = lmbd
    n = _data.size

    # solver = cp.ECOS
    solver = cp.CVXOPT
    reg_norm = order    # 1 = L1 trend filtering
                        # 2 = H-P filter

    # Form second difference matrix.
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)

    # Solve l1 trend filtering problem.
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(data - x) + vlambda * cp.norm(D @ x, reg_norm))
    prob = cp.Problem(obj)

    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=solver, verbose=True)
    print('Solver status: {}'.format(prob.status))

    # Check for error.
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
        return 0
    else:
        print("optimal objective value: {}".format(obj.value))
        return x.value

# Make an exception class for no sensors being valid, where I can add a message
class NoValidSensors(Exception):
    pass


class ThermalFrame:
    show_hexagons = False
    show_cb = False
    show_labels = False
    save_fig = False

    n_sensors = 64
    nx_sensor = 11
    ny_sensor = 5

    x_pcb = 410  # in mm
    y_pcb = 180  # in mm

    x_cells = 77
    y_cells = 19

    ## Bee arena size
    extent = x_min, x_max, y_min, y_max = [0, x_pcb, 0, y_pcb]

    grid = np.mgrid[x_min:x_max:1,y_min:y_max:1]

    ## Coordinates of each sensor (from Altium)
    sensor_x = np.array([
        21.7, 58.45, 95.225, 132, 168.75, 205.5, 242.25, 279, 315.775, 352.55, 389.3,
        21.7, 58.45, 95.225, 132, 168.75, 205.5, 242.25, 279, 315.775, 352.55, 389.3,
        21.7, 58.45, 95.225, 132, 168.75, 205.5, 242.25, 279, 315.775, 352.55, 389.3,
        21.7, 58.45, 95.225, 132, 168.75, 205.5, 242.25, 279, 315.775, 352.55, 389.3,
        21.7, 58.45, 95.225, 132, 168.75, 205.5, 242.25, 279, 315.775, 352.55, 389.3
    ])

    sensor_y = np.array([
        17.7, 17.7, 17.7, 17.7, 17.7, 17.7, 17.7, 17.7, 17.7, 17.7, 17.7,
        54.05, 54.05, 54.05, 54.05, 54.05, 54.05, 54.05, 54.05, 54.05, 54.05, 54.05,
        90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5, 90.5,
        126.925, 126.925, 126.925, 126.925, 126.925, 126.925, 126.925, 126.925, 126.925, 126.925, 126.925,
        163.35, 163.35, 163.35, 163.35, 163.35, 163.35, 163.35, 163.35, 163.35, 163.35, 163.35
    ])

    # Flip Y values to make the origin of coordinates on the lower left corner.
    # Otherwise images are represented with the origin on the upper left corner
    sensor_y = np.flipud(sensor_y)

    sensor_idx_map = np.array([60, 53, 54, 47, 40, 26, 27, 12, 13, 6, 7,
                               61, 52, 55, 46, 41, 25, 20, 16, 14, 5, 0,
                               62, 59, 48, 45, 42, 24, 21, 17, 15, 4, 1,
                               63, 58, 49, 44, 28, 31, 22, 18, 8, 11, 2,
                               56, 57, 50, 51, 29, 30, 23, 19, 9, 10, 3])

    ## Sensor 25 is already considered in the regular sensor array
    hdp_x = np.array([205.5, 202.925, 208.172, 200.3, 210.8, 197.675, 202.925, 208.172, 213.425])
    hdp_y = np.array([136.05, 131.5, 131.5, 126.925, 126.925, 122.386, 122.386, 122.386, 122.386])

    hdp_idx_map = np.array([32,
                            33, 34,
                            35, 36,
                            43, 37, 38, 39])

    mapping_t_to_row_col = {}
    for tgt, src in enumerate(sensor_idx_map):
        s = f"t{src:02d}"
        #tl = f"L{tgt:02d}"
        r = tgt // 11
        c = tgt % 11
        mapping_t_to_row_col[s] = {'row': r, 'col': c}

    def __init__(self, temperature_data, hive_id:int=-1, ts:pd.Timestamp=None, bad_sensors:list[int]=[]):
        '''
        temperature_data: list or np.ndarray of length 64 or 65 (if validity flag is included)
        hive_id: int, id of the hive (for plotting purposes and to check validity)
        ts: pd.Timestamp, timestamp of the data (to check validity)
        bad_sensors: list with indeces of bad sensors
        '''

        self.marker_size = 1
        self.thermal_field = None
        self.rbf_interpolator = None

        self.hive_id = hive_id
        self.ts = ts
        self.valid = True # Assume valid until proven otherwise
        if self.ts is not None and self.hive_id != -1:
            self.valid = valid_ts(self.ts, self.hive_id, recovery_time=180) # Thermal properties take longer to stabilise

        self.set_thermal_data(temperature_data)
        self.find_bad_sensors() # Check for any bad sensor

        if len(bad_sensors) > 0: # Manually add more bad sensors
            self.add_bad_sensors(bad_sensors)

    def set_thermal_data(self, temperatures):
        if len(temperatures) == ThermalFrame.n_sensors+1: # We have a validity flag at the end
            if temperatures[-1] == 'False' or temperatures[-1] == False:
                raise NoValidSensors('Not a single sensor has valid data!')
            else:
                temperatures = temperatures[:-1]
        if type(temperatures) == list:
            self.temperature_list = np.array(temperatures)
        elif type(temperatures) == np.ndarray:
            self.temperature_list = temperatures

        # Convert to float
        self.temperature_list = self.temperature_list.astype(float)
        
        self.temperature_array = self.temperature_list[ThermalFrame.sensor_idx_map] # Here the order is changed to match the sensor positions on the PCB

    def add_bad_sensors(self, bad_sensors:list[int]):
        '''Add more bad sensors'''
        self.bad_sensors = self.bad_sensors + bad_sensors
        self.updateBadSensors()

    def find_bad_sensors(self):
        '''Find the indices of bad sensors in a list or array of thermal data.
            For now only 2 cases are verified'''
        bad_sensors_idx = np.asarray((self.temperature_list==np.inf)|
                                     (self.temperature_list<=-273.0)|
                                     (self.temperature_list>2000)|
                                     (np.isnan(self.temperature_list))).nonzero()
        self.bad_sensors = bad_sensors_idx[0].tolist() # Ignore previously detected bad sensors
        self.updateBadSensors()

    def updateBadSensors(self):
        '''Update the bad sensors instance variables'''
        self.bad_sensors = list(set(self.bad_sensors)) # Remove duplicates
        self.bad_sensors.sort()

        if len(self.bad_sensors) >= ThermalFrame.n_sensors:
            raise NoValidSensors('Not a single sensor has valid data!')

        if len(self.bad_sensors) > 0:
            self.n_bad_sensors = len(self.bad_sensors)
            bad_sensors_array_idx = np.where(np.isin(ThermalFrame.sensor_idx_map, self.bad_sensors))[0]

            self.sensor_x_faulty,self.sensor_y_faulty = ThermalFrame.sensor_x[bad_sensors_array_idx], ThermalFrame.sensor_y[bad_sensors_array_idx]
            self.sensor_x_trusty,self.sensor_y_trusty = np.delete(ThermalFrame.sensor_x,bad_sensors_array_idx), np.delete(ThermalFrame.sensor_y,bad_sensors_array_idx)
            self.temp_array_trusty = np.delete(self.temperature_array, bad_sensors_array_idx)
        else:
            self.bad_sensors = ()
            self.n_bad_sensors = 0
            self.sensor_x_faulty = []
            self.sensor_y_faulty = []
            self.temp_array_trusty = self.temperature_array
            self.sensor_x_trusty = ThermalFrame.sensor_x
            self.sensor_y_trusty = ThermalFrame.sensor_y

    def draw_hexagons(self, ax):
        ## Dont plot this. Just to get the centers.
        c1 = ax.hexbin([], [],
                        gridsize=(self.x_cells, self.y_cells),
                        extent=(0.7, (self.x_pcb - ((self.x_pcb / (self.x_cells) + 0.3, 0.5, (self.y_pcb - 6.5))))),
                        # extent=(0, x_pcb, 0, y_pcb),
                        # cmap='Blues',
                        linewidths=0.5,
                        edgecolors='k',
                        # alpha=0.1,
                        facecolor=None,
                        visible=False,
                        zorder=-1)

        ## Adjust hexagons offsets
        ofs = c1.get_offsets()
        xo, yo = (self.x_pcb / self.x_cells) / 2, 3.5
        # xo, yo = 0.0, 0.0
        ofs[:, 0] += xo
        ofs[:, 1] += yo
        c1.set_offsets(ofs)

        ## hexagons centers
        hex_cen = c1.get_offsets()
        # hex_x, hex_y = zip(*hex_cen)
        return hex_cen

    # TODO: these methods seem to get the Y coordinate bad. FIXME!!! 13-03-2024 RMM
    ''' using with demo data
    tu.ThermalFrame.get_max_sensor_pos(temps_np)
    tu.ThermalFrame.get_mm_pos_of_sensor(2,1)
    x,y = tu.ThermalFrame.get_mm_pos_of_max_sensor(temps_np)
    plt.plot(x, y, marker='o', )
    '''
    
    def get_max_sensor_pos(self, temps:np.ndarray):
        '''Return the position of maximum temperature in the raw 64 temp array
        
        return a tuple (row, col) [row in 0..4, col in 0..10]
        '''
        # make all the columns for HDP sensors to be -273 so they don't be the max value
        t_no_hdp = temps.copy()
        t_no_hdp[ThermalFrame.hdp_idx_map] = -273.0

        i = np.argmax(t_no_hdp)
        i_str = f"t{i:02d}"

        rv = self.mapping_t_to_row_col.get(i_str, None)
        if rv is None:
            raise RuntimeError(f"Could not find sensor {i_str} in mapping")

        return rv
    
    def get_mm_pos_of_sensor(self, row:int, col:int):
        '''Return the position of the sensor in mm'''
        return (self.sensor_x[col], self.sensor_y[row])

    def get_mm_pos_of_max_sensor(self, temps:np.ndarray):
        '''Return the position of the sensor in mm'''
        rc_dict = self.get_max_sensor_pos(temps)
        x, y = self.get_mm_pos_of_sensor(rc_dict['row'], rc_dict['col'])
        return (x, y)

    def calculate_thermal_field(self,verbose=False):
        sensor_pos = np.vstack([self.sensor_x_trusty, self.sensor_y_trusty]).T
        if verbose:
            print(f"shape trusty sensor_pos: {np.shape(sensor_pos)}")
            print(f"shape temp_array_trusty: {np.shape(self.temp_array_trusty)}")

        self.rbf = RBFInterpolator(sensor_pos, self.temp_array_trusty, kernel='linear')
        grid_flattened = ThermalFrame.grid.reshape(2, -1).T
        if verbose:
            print(ThermalFrame.grid)
            print(grid_flattened)
            print(f"grid_flattened: {np.shape(grid_flattened)}")
            print(f"grid: {np.shape(ThermalFrame.grid)}")
        sensor_zi = self.rbf(grid_flattened)
        ygrid = sensor_zi.reshape(410, 180).T
        ygrid = np.flipud(ygrid) # Flip the y-axis to match the sensor positions
        if verbose:
            print(f"ygrid: {np.shape(ygrid)}")
            print(f"ygrid: {ygrid}")
        
        self.thermal_field = ygrid
        self.max_temp = np.max(self.thermal_field)
        self.min_temp = np.min(self.thermal_field)

        return self.thermal_field

    def plot_thermal_field(self, ax, cmap=None, 
                           show_cb:bool=True, 
                           show_sensors:bool=False, 
                           show_max_temp:bool=False, 
                           contours:list=None, annotate_contours:bool=True, 
                           v_min=None, v_max=None, 
                           viewed_from:str = 'front'):
        
        assert viewed_from in ['front', 'back'], "viewed_from must be 'front' or 'back'"
        cm = plt.get_cmap('bwr') if cmap is None else cmap

        if self.thermal_field is None:
            temp_field = self.calculate_thermal_field()
        else:
            temp_field = self.thermal_field.copy()

        if viewed_from == 'back':
            # Flip the thermal field horizontally
            temp_field = np.fliplr(temp_field)

        # Plot temperature field
        _image = ax.imshow(temp_field,extent=ThermalFrame.extent,cmap=cm, vmin=v_min, vmax=v_max)

        ax.yaxis.set_major_locator(plt.MaxNLocator(3)) # Reduce the number of ticks on the y-axis

        if show_cb:
            cbar = plt.colorbar(_image, ax=ax, orientation='vertical')
            cbar.ax.set_position([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.04, ax.get_position().height])  # Adjust position relative to ax

        # Mark sensors
        if show_sensors:
            _x_trusty = self.sensor_x_trusty if viewed_from == 'front' else self.x_pcb - self.sensor_x_trusty
            _x_faulty = self.sensor_x_faulty if viewed_from == 'front' else self.x_pcb - self.sensor_x_faulty
            # Valid sensors (green squares)
            ax.scatter(_x_trusty, self.sensor_y_trusty, s=self.marker_size**2, c='g', marker='s', linewidths=0.5,label='Valid sensors')
            # Mark faulty sensors (with a cross)
            ax.scatter(_x_faulty, self.sensor_y_faulty, s=self.marker_size*20, c='m', marker='x', linewidths=2,label='Faulty sensors')

        # Contour lines
        if contours is not None:
            cs = ax.contour(temp_field, levels=contours, colors='k', linewidths=0.3, origin='upper') # The thermal field is stored with origin upper (numpy arrays)
            if annotate_contours:
                ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f"{x:.0f} C")

        if show_max_temp:
            x, y = self.get_max_temp_pos()
            if viewed_from == 'back':
                x = self.x_pcb - x
            ax.plot(x, y, 'go', markersize=4,label='Max temp')
            ax.annotate(f'{self.max_temp:.1f}°C', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=10, color='black')
            ax.legend(loc='upper right', fontsize=7)

        ax.set_title(f'Thermal field for hive {self.hive_id} {"(no valid sensors)" if self.n_bad_sensors==ThermalFrame.n_sensors else ""}')
        ax.set_xlim([0, self.x_pcb])
        ax.set_ylim([0, self.y_pcb])

        if not self.valid:
            # Add a red transparent rectangle to indicate invalid data, and write "Invalid dt" in the middle in red too
            ax.add_patch(plt.Rectangle((0, 0), self.x_pcb, self.y_pcb, color='red', alpha=0.5))
            # Put a white rectangle behind the text to make it more visible
            ax.text(self.x_pcb/2, self.y_pcb/2, 'Invalid dt', color='black', fontsize=40, ha='center', va='center', alpha=1)

        return ax

    def set_clevels(self, c_levels):
        self.c_levels = c_levels

    def get_max_temp_pos(self, origin:str= 'lower', verbose=False):
        '''Return the position (x-y)
        params:
        - origin : 'lower' or 'upper' (default is 'lower', with the origin bottom left) of maximum temperature in the thermal field
        '''
        assert origin in ['lower', 'upper'], "Origin must be 'lower' or 'upper'"
        if verbose:
            print(f"shape thermal field: {np.shape(self.thermal_field)}")
            print(ThermalFrame.y_pcb)
        i_max = np.argwhere(self.thermal_field == np.max(self.thermal_field))[0]
        if verbose:
            print(f"i_max: {i_max}")

        if origin == 'lower':
            return (i_max[1], ThermalFrame.y_pcb-i_max[0])
        else:
            return (i_max[1], i_max[0])