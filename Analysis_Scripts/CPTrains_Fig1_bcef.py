import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyart # radar plotting
import numpy as np
from datetime import timedelta
from datetime import datetime
import datetime
import pandas as pd
import netCDF4 as nc
import glob # for using wildcards etc.
import cartopy.crs as ccrs
import cartopy.feature as feat
import cartopy
from pyart.graph import cm
import pytz
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# met plotting libraries
from metpy.calc import wind_components
from metpy.plots import StationPlot, StationPlotLayout, simple_layout
from metpy.units import units
from netCDF4 import Dataset
from matplotlib.colors import LinearSegmentedColormap

# for interfacing with AWS
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
# libraries for parallelization
from jug import TaskGenerator
import jug
# set some plotting info
import matplotlib 
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
style = "/home/cneumaie/Styles/christine-paperlight.mpstyle"

plt.style.use(style)
matplotlib.rc('xtick', labelsize=32) 
matplotlib.rc('ytick', labelsize=32) 
matplotlib.rc('font', size=32) 

def daterange(start_date, end_date):
    for n in range(int(((end_date.date()) - (start_date.date())).days+1)):
        yield start_date.date() + datetime.timedelta(n)
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    try:
        plt.cm.get_cmap(name)
    except ValueError:
        plt.register_cmap(cmap=newcmap)

    return newcmap
def convert_gr_table(grstr):
    '''
    Convert a color table designed for GRLevel2/3 to a python one.
    Be sure that the min/max values are identical. 
    '''
    spstr = grstr.split("color:")
    spstr = [x.strip() for x in spstr]
    varvalues = list()
    red1values = list()
    red2values = list()
    blue1values = list()
    blue2values = list()
    green1values = list()
    green2values = list()

    for interval in spstr:
        if interval == '':
            continue
        indivals = interval.split()
        varvalues.append(int(indivals[0]))
        red1values.append(int(indivals[1]))
        green1values.append(int(indivals[2]))
        blue1values.append(int(indivals[3]))
        if len(indivals)<5:
            #we aren't discontinuous here.
            red2values.append(-1)
            green2values.append(-1)
            blue2values.append(-1)
        else:
            red2values.append(int(indivals[4]))
            green2values.append(int(indivals[5]))
            blue2values.append(int(indivals[6]))
    
    normvarvals = [(x+(0-min(varvalues)))/
                   (max(varvalues)+(0-min(varvalues))) for x in varvalues]
    red1values = [x/255.0 for x in red1values]
    red2values = [x/255.0 for x in red2values]
    green1values = [x/255.0 for x in green1values]
    green2values = [x/255.0 for x in green2values]
    blue1values = [x/255.0 for x in blue1values]
    blue2values = [x/255.0 for x in blue2values]
    redvals = list()
    greenvals = list()
    bluevals = list()
    for i, num in enumerate(normvarvals):
        if i == 0:
            redvals.append((num, 0.0, red1values[i]))
            greenvals.append((num, 0.0, green1values[i]))
            bluevals.append((num, 0.0, blue1values[i]))
        
        else:
            if red2values[i-1]<0:
                redvals.append((num, red1values[i], red1values[i]))
                greenvals.append((num, green1values[i], green1values[i]))
                bluevals.append((num, blue1values[i], blue1values[i]))

            else:
                redvals.append((num, red2values[i-1], red1values[i]))
                greenvals.append((num, green2values[i-1], green1values[i]))
                bluevals.append((num, blue2values[i-1], blue1values[i]))

    cmapdict = {
        'red':tuple(redvals),
        'green':tuple(greenvals),
        'blue':tuple(bluevals)
    }
    return cmapdict
#@TaskGenerator    
def plot_radar_data(radar_file,terrain_data,i,fig):

    filename=radar_file
    print(filename)





    wdtbtable = convert_gr_table(grctable)
    try:
        plt.cm.get_cmap('wdtbtable')
    except ValueError:
        wdbt = LinearSegmentedColormap('wdtbtable', wdtbtable)
        plt.register_cmap(cmap=wdbt)


   

    # fig = plt.figure(figsize=figuresize)

    proj = ccrs.LambertConformal(central_longitude = cent_lon, central_latitude = cent_lat
                                    ,standard_parallels=[35])

    ax2 = fig.add_subplot(2,2,i+1, projection = proj)
   #ax2 = fig.add_subplot(2, 2, i, projection=proj)

    ax2.set_extent((lonmin, lonmax, latmin, latmax))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    # ax2.gridlines(draw_labels=True, crs=proj)
    # gl.xlocator = MultipleLocator(0.5)  # Set x-axis gridline locator to 0.5 degrees
    # gl.ylocator = MultipleLocator(0.5)  # Set y-axis gridline locator to 0.5 degrees

    # # Add some various map elements to the plot to make it recognizable
    radar = pyart.io.read_nexrad_archive(filename)
    display = pyart.graph.RadarMapDisplay(radar)
 


    # else:
    if var_to_plot == 'reflectivity':
        display.plot_ppi_map(var_to_plot, sweep=sweep, vmin=vmin, vmax=vmax, ax=ax2,         
                            mask_outside=True, cmap = 'wdtbtable', ticks=np.arange(vmin,vmax+1,10),
                            colorbar_flag = False
        )
    elif var_to_plot == 'velocity':
            display.plot_ppi_map(var_to_plot, sweep=sweep, vmin=vmin, vmax=vmax, ax=ax2,         
                            mask_outside=True, ticks=np.arange(vmin,vmax+1,10) 
        )
   


    # Get relevant shapefiles
    counties = cartopy.io.shapereader.Reader(county_shapefile)


    ax2.add_feature(cfeature.STATES, linewidth=1)
 
    ax2.add_geometries(counties.geometries(), ccrs.PlateCarree(),
                         edgecolor='grey', facecolor='None', linewidth=0.5)


    radar_time = nc.num2date(radar.time['data'][0], radar.time['units'],
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
    shrunk_cmap = shiftedColorMap(matplotlib.cm.terrain, start=0.3, midpoint=0.5, stop=0.95, name='shrunk_terrain')


    # tercont = ax2.contourf(terrain_data['x'], terrain_data['y'], terrain_data['z'], 
    #             levels=np.linspace(0,2750,100), cmap=shrunk_cmap, zorder=0, extend='both',
    #             transform_first=True)



    # Plot primary site
    
    # ax2.scatter(main_loc[1], main_loc[0], transform=ccrs.PlateCarree(), **main_loc_plot_opts) # Uncomment only for panel a)



    curr_local_time = radar_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=timezone)
    if timezone == mountain_tz:
        timezone_label = "MT"
    elif timezone == central_tz:
        timezone_label = "CT"
    elif timezone == eastern_tz:
        timezone_label = "ET"
    ax2.set_title(title_firstpart+"\n"+
                r""+curr_local_time.strftime("%Y/%m/%d %H:%M:%S ")+timezone_label, size=32)
    ax2.text(.01, 1.1, title_prefix, ha='left', va='top', weight = 'bold', size = 50, transform=ax2.transAxes)

    # if plot_elev_colorbar == True:
        # cbaxes = fig.add_axes([0.2, 0.02, 0.6, 0.02])   # This is the position for the colorbar
        # cb = fig.colorbar(tercont, cax =cbaxes, ticks=np.arange(0,2751,500), orientation='horizontal')
        # cbaxes.yaxis.set_ticks(np.arange(0,2500,250))
        # cbaxes.yaxis.set_ticks_position('none')
        # cb.set_label('Elevation (m)')


    display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0], color = "black", markersize = 10)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # plt.savefig(plot_dir+
    #     plot_prefix+"_"+radar_time.strftime("%y%m%d_%H%M%S")+
    #     ".png", dpi=600, bbox_inches="tight")

    # plt.savefig(plot_dir+
    #     plot_prefix+radar_time.strftime("%y%m%d_%H%M%S")+
    #     ".png", dpi=600, bbox_inches="tight")

    # plt.close(fig)
    # if i==3:
    # return tercont
# @TaskGenerator
def get_radar_data(radarname, start_datetime, end_datetime, download_dir):
    '''gets the radar data from an AWS bucket
    '''
    s3 = boto3.resource('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    nexrad_bucket = s3.Bucket('noaa-nexrad-level2')

    # get all radar bucket objects
    all_radar_data = list()
    for curr_date in daterange(start_datetime, end_datetime):
        #print(curr_date)
        curr_prefix = curr_date.strftime("%Y/%m/%d/")+radarname
        curr_radar_data = list(nexrad_bucket.objects.filter(Prefix=curr_prefix))
        radar_dates = [datetime.datetime.strptime(in_obj.key.split('/')[-1].split('V06')[0], radarname+"%Y%m%d_%H%M%S_") for in_obj in curr_radar_data]
        dates_in_range_sel = np.logical_and(np.array(radar_dates)>start_datetime, np.array(radar_dates)<end_datetime)
        all_radar_data+=np.array(curr_radar_data)[dates_in_range_sel].tolist()

    # Download radar data from AWS
    radfiles = list()
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    for i, curr_rad_obj in enumerate(all_radar_data):
        radar_out_filename = download_dir+curr_rad_obj.key.split('/')[-1]
        if 'MDM' in radar_out_filename:
            continue

        # if we have already downloaded it, don't download it again. 
        if os.path.exists(radar_out_filename):
            pass
        else:
            print("Downloading "+ curr_rad_obj.key.split('/')[-1])
            nexrad_bucket.download_file(Key=curr_rad_obj.key, Filename=radar_out_filename)
        radfiles.append(radar_out_filename)

    return radfiles
def get_terrain_data(topography_file, projection, lonmin, lonmax, latmin, latmax, pad_deg=0.5):
    '''
    Load in the terrain data and convert the lon coordinates to projection coordinates
    '''
    #load in topo dataset and plot
    topods = nc.Dataset(topography_file)
    #these values need to be padded to avoid cutting off the terrain on the edges
    pad_deg = 0.5
    padlonmin = lonmin-pad_deg
    padlonmax = lonmax+pad_deg
    padlatmin = latmin-pad_deg
    padlatmax = latmax+pad_deg


    terlonmin = np.argmin(np.abs(np.array(topods.variables['x'])-(padlonmin)))
    terlonmax = np.argmin(np.abs(np.array(topods.variables['x'])-(padlonmax)))
    terlatmin = np.argmin(np.abs(np.array(topods.variables['y'])-(padlatmin)))
    terlatmax = np.argmin(np.abs(np.array(topods.variables['y'])-(padlatmax)))
    llon, llat = np.meshgrid(topods.variables['x'][terlonmin:terlonmax]
                             , topods.variables['y'][terlatmin:terlatmax])
    all_pts = projection.transform_points(ccrs.PlateCarree(), llon, llat)
    x = all_pts[:,:,0]
    y = all_pts[:,:,1]
    return x, y, np.array(topods.variables['z'][terlatmin:terlatmax, terlonmin:terlonmax])

# User options
mountain_tz = pytz.timezone('America/Denver') #for KCYS and KGLD
central_tz = pytz.timezone('America/Chicago') #for KGDX
eastern_tz = pytz.timezone('America/New_York') #for KRAX
# Radar code

# plotting options
figuresize=[30, 20]
fig = plt.figure(figsize = figuresize)
for i in range(4):
    if i == 0:
        radarname = 'KCYS'
        cent_lat = 40
        cent_lon = -104
        start_datetime = datetime.datetime(2024,8,19,21,41) #panel b
        end_datetime = datetime.datetime(2024,8,19,21,42)
        #Latitude/Longitude boundaries
        lonmin = -105.5
        lonmax = -104
        latmin = 40.5
        latmax = 41.5
        plot_elev_colorbar = False
        timezone = mountain_tz
        title_prefix = 'b)'
    elif i==1:    
        radarname = 'KGLD' 
        cent_lat = 40
        cent_lon = -104
        lonmin = -102.5
        lonmax = -100.5
        latmin = 38.5
        latmax = 40
        start_datetime = datetime.datetime(2023,6,5,23,1) # panel c
        end_datetime = datetime.datetime(2023,6,5,23,2)
        plot_elev_colorbar = False
        timezone = mountain_tz
        title_prefix = 'c)'
    elif i ==2:
        radarname = 'KDGX'
        cent_lat = 32
        cent_lon = -90
        lonmin = -90
        lonmax = -90+1.5
        latmin = 32.5-1
        latmax = 32.5
        plot_elev_colorbar = False
        timezone = central_tz
        start_datetime = datetime.datetime(2023,6,8,23,23) # panel e
        end_datetime = datetime.datetime(2023,6,8,23,24)
        title_prefix = 'e)'
    elif i ==3:
        radarname = 'KRAX'
        cent_lat = 35.5
        cent_lon = -78.5
        lonmin = -79.8
        lonmax = -78.2
        latmin = cent_lat-0.5
        latmax = cent_lat+0.5
        start_datetime = datetime.datetime(2022,7,16,19,1) # panel f
        end_datetime = datetime.datetime(2022,7,16,19,2)
        title_prefix = 'f)'
        plot_elev_colorbar = False
        timezone = eastern_tz
    
    # Radar variable to plot, options: 'reflectivity', 'velocity', 'composite_reflectivity'
    var_to_plot = 'reflectivity'
    # var_to_plot = 'velocity'
    #var_to_plot = 'composite_reflectivity'

    # print("test")
    topography_file = './map_topo_files/ETOPO1_Ice_g_gmt4.grd'

    county_shapefile = './map_topo_files/cb_2021_us_county_5m/cb_2021_us_county_5m.shp'
    # start_datetime = datetime.datetime(2022, 7, 16, 16)
    # end_datetime = datetime.datetime(2022, 7, 16,21) 
    download_dir='./'+radarname+"_radar_data/"

    #plot_dir = './/'
    # plot_dir: Where do you want the plots to output to#   If this directory doesn't exist, the script will create it.
    # plot_prefix: what prefix to add to the output file names
    if var_to_plot == 'reflectivity': 
        plot_dir = './'+radarname+"_plots_refl_Fig1/"
        plot_prefix = radarname+'_ref05'
    elif var_to_plot == 'velocity': 
        plot_dir = './'+radarname+"_plots_vel/"
        plot_prefix = radarname+'_vel05'
    # Download roads shapefiles from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
    # primary and secondary roads by state

    # CO roads shapefile
    co_roads_shapefile = './map_topo_files/tl_2021_08_prisecroads/tl_2021_08_prisecroads.shp'
    # WY roads shapefile
    wy_roads_shapefile = './map_topo_files/tl_2021_56_prisecroads/tl_2021_56_prisecroads.shp'
    # NE roads shapefile
    ne_roads_shapefile = './map_topo_files/tl_2021_31_prisecroads/tl_2021_31_prisecroads.shp'
    # State border shapefile
    # state_shapefile = './map_topo_files/tl_2022_us_state/tl_2022_us_state.shp'

    all_road_shapefiles = [co_roads_shapefile, wy_roads_shapefile, ne_roads_shapefile]






    # maximum time away from plot time that the metar station plot can be
    max_station_time = timedelta(hours=1)
    #max_station_time = timedelta(minutes=5)

    # whether or not to include the elevation colorbar
    #plot_elev_colorbar = True ##True normally
    # reflectivity color table
    grctable_ref = """color: -30 116 78 173 147 141 117
    color: -20 150 145 83 210 212 180
    color: -10 204 207 180 65 91 158
    color: 10 67 97 162 106 208 228
    color: 18 111 214 232 53 213 91
    color: 22 17 213 24 9 94 9
    color: 35 29 104 9 234 210 4 
    color: 40 255 226 0 255 128 0
    color: 50 255 0 0 113 0 0
    color: 60 255 255 255 255 146 255
    color: 65 255 117 255 225 11 227
    color: 70 178 0 255 99 0 214
    color: 75 5 236 240 1 32 32
    color: 85 1 32 32
    color: 95 1 32 32"""
    # title_firstpart: What you want the first line of the title to be
    # vmin, vmax: minimum and maximum values to plot
    #   make sure that these match the color table
    if var_to_plot == 'reflectivity':
        title_firstpart = r""+radarname+r" Reflectivity 0.5$^\circ$ "
        vmin, vmax = -30, 95
        grctable = grctable_ref 
        sweep = 0
    elif var_to_plot == 'velocity':
        title_firstpart = r""+radarname+r" Velocity 0.5$^\circ$ "
        vmin, vmax = -70, 70
        grctable = grctable_ref 
        sweep = 0
    # location of main location (e.g., CPER) in (latitude, longitude) format
    # Set this to None to not plot it. 
    main_loc = (40.80985556259899, -104.7782733197491) # this is SGRC
    # options for plotting the main location point
    main_loc_plot_opts = {
        'marker': '*',
        'facecolor': 'k',
        'edgecolor': 'k',
        's': 250
    }


    radfiles = get_radar_data(radarname, start_datetime, end_datetime, download_dir)

    proj = ccrs.LambertConformal(central_longitude = cent_lon, central_latitude = cent_lat
                                ,standard_parallels=[35])

    terrain_x, terrain_y, terrain_data = get_terrain_data(topography_file, projection=proj,
                                            lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)
    terrain_data = {'x': terrain_x, 'y':terrain_y, 'z': terrain_data}
# plot_radar_data(terrain_data=terrain_data)
    for radfilenum, filename in enumerate((radfiles)):
    # for radfilenum, filename in enumerate((radfiles)):
        tercount = plot_radar_data(filename, terrain_data=terrain_data,i=i, fig = fig)
wdtbtable = convert_gr_table(grctable_ref)
reflectivity_cmap = LinearSegmentedColormap('wdtbtable', wdtbtable)

# Adjust and add colorbars
# cbaxes = fig.add_axes([0.2, 0.02, 0.6, 0.02]) 
# Reflectivity colorbar (right side)
cax_reflectivity = fig.add_axes([0.9, 0.14, 0.02, 0.7])  # [left, bottom, width, height]
cbar_reflectivity = fig.colorbar(
    plt.cm.ScalarMappable(cmap=reflectivity_cmap, norm=plt.Normalize(vmin, vmax)),
    cax=cax_reflectivity, orientation='vertical', label="Reflectivity (dBZ)"
)
# Elevation color map (example)
shrunk_cmap = LinearSegmentedColormap.from_list("shrunk_terrain", ["#d9f0d3", "#a6dba0", "#008837"], N=100)
# cbar_elevation = fig.colorbar(
#     plt.cm.ScalarMappable(cmap=shrunk_cmap, norm=plt.Normalize(0, 2750)),
#     ax=axs[1, :], orientation='horizontal', label="Elevation (m)"
# )
# cbaxes = fig.add_axes([0.2, 0.02, 0.6, 0.02])
# Elevation colorbar (bottom)
# cax_elevation = fig.add_axes([0.2, -0.01, 0.6, 0.02])  # [left, bottom, width, height] 
# cbar_elevation = fig.colorbar(
#     tercount, ticks = np.arange(0, 2750,500),
#     cax =cax_elevation, orientation='horizontal', label="Elevation (m)"
# )
# plt.tight_layout(rect=[0, 0.1, 0.9, 1.0])  # Adjust layout to fit colorbars
plt.savefig(plot_dir+
    plot_prefix+"test_fig1"+
    "v2.png", dpi=600, bbox_inches="tight")

plt.close(fig)