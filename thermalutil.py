'''
Author(s):
    Cyril Monette, EPFL, cyril.monette@epfl.ch
'''

import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

def preview(df_therm_data,show_sensors=False,vmin=None,vmax=None,rows=None):
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
    plt.figure(); plt.clf()
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 7))
    n_data_points = df_therm_data.shape[0]
    if rows is None:
        rows = np.linspace(0, n_data_points-1, 6, dtype=int)

    coords = []
    print("=====================================================================================================================================================")
    print("=====================================================================================================================================================")
    print(f"Previewing thermal data:")
    for i, row in enumerate(rows):
        temps = df_therm_data.iloc[row]
        temps_np = temps.to_numpy()
        try:
            thermalframe = ThermalFrame(temperature_data=temps_np, show_sensors=show_sensors)

        except NoValidSensors as e:
            print(f"Skipping row {row} due to no valid sensors")
            continue
        
        temp_field = thermalframe.calculate_thermal_field(verbose=False)

        x, y = thermalframe.get_max_temp_pos()
        coords.append((row, df_therm_data.iloc[row].name, x, y))

        print(f"Max temp at {x}, {y} for row {row} [tmax = {np.max(temp_field)}]")
        a = ax.flat[i]
        thermalframe.plot_thermal_field(a,show_cb=True,v_min=vmin,v_max=vmax)
        a.plot(x, y, 'go', markersize=4,label='Max temp')
        a.set_title(f"Row {row} -> {str(df_therm_data.iloc[row].name)}")
        a.legend(loc='upper right', fontsize=7)

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
    # show_contour = True
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

    ## Flip Y values to align make the origin on the lower left corner
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

    def __init__(self, temperature_data, hive_id=None, hive_pos=None, bad_sensors=None,show_sensors=False,show_contour=True):
        '''
        bad_sensors: list with indeces of bad sensors
        '''

        self.marker_size = 1
        self.thermal_field = None
        self.rbf_interpolator = None

        self.hive_id = hive_id
        self.hive_pos = hive_pos

        self.show_sensors = show_sensors
        self.show_contour = show_contour

        self.image = None # Store the imshow object

        self.set_thermal_data(temperature_data)
        self.find_bad_sensors() # Check for any bad sensor

        if bad_sensors is not None: # Manually add more bad sensors
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
        
        self.temperature_array = self.temperature_list[ThermalFrame.sensor_idx_map] # Here the order is changed to match the sensor positions on the PCB

    def add_bad_sensors(self, list_bad_sensors):
        '''Add more bad sensors'''
        self.bad_sensors = np.concatenate((self.bad_sensors, list_bad_sensors))
        self.n_bad_sensors = len(self.bad_sensors)

    def find_bad_sensors(self):
        '''Find the indices of bad sensors in a list or array of thermal data.
            For now only 2 cases are verified'''
        bad_sensors_idx = np.asarray((self.temperature_list==np.inf)|(self.temperature_list==-273.0)|(self.temperature_list>2000)).nonzero()

        if bad_sensors_idx[0].shape[0] ==ThermalFrame.n_sensors:
            raise NoValidSensors('Not a single sensor has valid data!')

        if bad_sensors_idx[0].shape[0] > 0:
            self.bad_sensors = bad_sensors_idx[0]
            self.n_bad_sensors = len(bad_sensors_idx[0])
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

        return self.thermal_field

    def plot_thermal_field(self, ax, cmap=None, show_cb=False, contours=None, v_min=None, v_max=None):
        cm = plt.get_cmap('bwr') if cmap is None else cmap

        if self.thermal_field is None:
            temp_field = self.calculate_thermal_field()
        else:
            temp_field = self.thermal_field.copy()

        # Plot temperature field
        self.image = ax.imshow(temp_field,extent=ThermalFrame.extent,cmap=cm, vmin=v_min, vmax=v_max)

        ax.yaxis.set_major_locator(plt.MaxNLocator(3)) # Reduce the number of ticks on the y-axis

        if show_cb:
            cbar = plt.colorbar(self.image, ax=ax, orientation='vertical')
            cbar.ax.set_position([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.04, ax.get_position().height])  # Adjust position relative to ax

        # Mark sensors
        if self.show_sensors:
            ax.plot(self.sensor_x_trusty, self.sensor_y_trusty, 's', ms=self.marker_size, c='g',label='Valid sensors')
            # Mark faulty sensors (with a cross)
            ax.scatter(self.sensor_x_faulty, self.sensor_y_faulty, s=self.marker_size*20, c='m', marker='x', linewidths=2,label='Faulty sensors')

        # Contour lines
        if contours is not None:
            cs = ax.contour(temp_field, levels=contours, colors='k', linewidths=0.3)
            # Flip the contours along the y-axis
            for collection in cs.collections:
                for path in collection.get_paths():
                    path.vertices[:, 1] = ThermalFrame.y_pcb - path.vertices[:, 1]

        ax.set_xlim([0, self.x_pcb])
        ax.set_ylim([0, self.y_pcb])

        return ax

    def set_clevels(self, c_levels):
        self.c_levels = c_levels

    def get_max_temp(self):
        return np.max(self.thermal_field)

    def get_max_temp_pos(self,verbose=False):
        '''Return the position (x-y with the origin bottom left) of maximum temperature in the thermal field'''
        if verbose:
            print(f"shape thermal field: {np.shape(self.thermal_field)}")
        i_max = np.argwhere(self.thermal_field == np.max(self.thermal_field))[0]

        return (i_max[1], ThermalFrame.y_pcb-i_max[0])