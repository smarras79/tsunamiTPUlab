"""

Library for GeoClaw plotting

Created by Adrian Santiago Tate (adrianst@stanford.edu)
	October, 2018, to January 2019

Edited by Ian Madden (iamadden@stanford.edu)
      April 2023

To Do:
- Automaticaly create movies

"""

# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import mplot3d
from copy import copy


class geoData:
	"""
	Class for reading and plotting GeoClaw output

	-	User may specify a normalized start and end time to reduce computation
	"""


	def __init__(self, pathOutput = None, pathTopo = None, startTime = 0, endTime = 100):

		if pathOutput is not None:

			#	Path to GeoClaw output
			self.pathOutput = pathOutput

			#	Get list of all GeoClaw fort.q and fort.t output files in pathOutput directory
			fortQFileList = self.getFortQFilelist()
			fortTFileList = self.getFortTFilelist()

			#	Number of files, i.e. "time frames"
			nFrames = len(fortQFileList)

			#	Normalize start and end times by number of fort.q files
			if startTime is not 0:
				startFrame = int(np.floor(startTime/100 * nFrames))
			else:
				startFrame = 0
			if endTime is not 100:
				endFrame = int(np.floor(endTime/100 * nFrames))
			else:
				endFrame = nFrames

			#	Slice from list of all output files to select data from time of interest
			self.fortQFileList = fortQFileList[startFrame : endFrame]
			self.fortTFileList = fortTFileList[startFrame : endFrame]

			#	Create dictionary with metadata and data
			self.dataDict = self.getDataDict()

			#	Create list with times for each time frame from fort.f files
			self.time = self.getTime()

			#	Create pointcloud from metadata
			print('Creating point cloud...')
			self.createPointCloud()

		else:
			print('No path specified to directory with output data. Try again.')
		
		#	Read topography file and save metadata and topoData to dataframe
		if pathTopo is not None:

			#	Path to GeoClaw topotype 2 file
			self.pathTopo = pathTopo

			#	Read topo file
			print('Reading topo file...')
			self.topo = self.getTopo()

		else:
			print('No path specified to directory with topo data. Try again.')

		print("Done reading data.")


	def getFortTFilelist(self):
		"""
		Returns sorted list of all files in given directory that contain a "fort.t" string in filename
		"""

		fileList = [i for i in os.listdir(self.pathOutput) if os.path.isfile(os.path.join(self.pathOutput,i)) and 'fort.t' in i]

		return sorted(fileList)


	def getFortQFilelist(self):
		"""
		Returns sorted list of all files in given directory that contain a "fort.q" string in filename
		"""

		fileList = [i for i in os.listdir(self.pathOutput) if os.path.isfile(os.path.join(self.pathOutput,i)) and 'fort.q' in i]

		return sorted(fileList)


	def readFortTFile(self, fortTFilename):
		"""
		Low-level reading of fort.t files.

		Returns time in seconds for given output file.
		"""
		
		#	Open file, remove blank lines
		with open(self.pathOutput + fortTFilename) as myFile:
			dataLines = removeBlankLines(myFile)

			#	Find line with time
			for line in dataLines:
				if "time" in line:
					return float(line[:18])


	def readFortQFile(self,fortqFilename):
		"""
		Low-level reading of fort.q files. 

		Returns dictionary with data and metadata dataframes.
		"""

		# Initialize lists for data
		lineDy = []
		gridNumber = []
		amrLevel = []
		mx = []
		my = []
		xlow = []
		ylow = []
		dx = []
		dy = []
		data = []
		gridCounter = -1

		#	Open file, remove blank lines
		with open(self.pathOutput + fortqFilename) as myFile:
			dataLines = removeBlankLines(myFile)

			#	Read lines, look for data
			for num, line in enumerate(dataLines, 1):
				if 'grid_number' in line:
					gridNumber.append(int(line[:20]))
					gridCounter += 1
				elif 'AMR_level' in line:
					amrLevel.append(int(line[:20]))
				elif 'mx' in line:
					mx.append(int(line[:20]))
				elif 'my' in line:
					my.append(int(line[:20]))
				elif 'xlow' in line:
					xlow.append(float(line.strip()[:22]))
				elif 'ylow' in line:
					ylow.append(float(line.strip()[:22]))
				elif 'dx' in line:
					dx.append(float(line.strip()[:22]))
				elif 'dy' in line:
					lineDy.append(num)
					dy.append(float(line.strip()[:22]))
				elif num > lineDy[gridCounter]:
					data.append([float(i) for i in line.split()])

		#	Transpose lists into dataframe format
		metadata = list(map(list, zip(*[gridNumber, amrLevel, mx, my, xlow, ylow, dx, dy])))

		#	Create dataframes
		metadata = pd.DataFrame(metadata, columns = ['gridNumber','amrLevel', 'mx', 'my', 'xlow', 'ylow', 'dx', 'dy'])
		data = pd.DataFrame(data, columns = ['q(1)','q(2)', 'q(3)', 'q(4)'])

		#	Add empty column to fill with zTopo values later
		data['zTopo'] = np.nan

		return {'metadata':metadata, 'data':data}


	def getDataDict(self):
		"""
		"""
		print("Reading {} fort.q files...".format(len(self.fortQFileList)))

		#	Initialize dictionary for dictionaries
		dataDict = {}

		#	Iteratively read fort.q files
		for timeFrameName in self.fortQFileList:
			timeFrame = self.readFortQFile(timeFrameName)
			dataDict[timeFrameName] = timeFrame

		return dataDict


	def getTime(self):
		"""
		"""
		print("Reading {} fort.t files...".format(len(self.fortTFileList)))

		#	Initialize list for times
		time = []

		#	Iteratevely read fort.t files
		for timeFrameIdx in range(len(self.fortQFileList)):
			time.append(self.readFortTFile(self.fortTFileList[timeFrameIdx]))
		
		return time


	def createPointCloud(self):
		"""
		Extracts x and y values from metadata and adds pointcloud to data dataframes
		"""

		#	Iterate through time frames
		for timeFrameName in self.fortQFileList:
			metadata = self.dataDict[timeFrameName]['metadata']

			x = []
			y = []

			#	Iterate through grids
			for grid in range(len(metadata['gridNumber'])):

				#	Define variables
				xLowBound = metadata['xlow'][grid]
				yLowBound = metadata['ylow'][grid]
				dx = metadata['dx'][grid]
				dy = metadata['dy'][grid]
				mx = metadata['mx'][grid]
				my = metadata['my'][grid]

				#	Create lists of x and y values
				xValues = list(np.linspace(xLowBound, xLowBound + mx * dx, mx))
				yValues = list(np.linspace(yLowBound, yLowBound + my * dy, my))

				# format x and y values to match fortran output
				x.append(xValues * my)
				ygrid = []
				for yValue in yValues: ygrid.append([yValue] * mx)
				y.append([item for sublist in ygrid for item in sublist])
				if len(xValues * my) != len([item for sublist in ygrid for item in sublist]):
					print('problem')

			#	Flatten lists of lists
			x = [item for sublist in x for item in sublist]
			y = [item for sublist in y for item in sublist]

			#	Add to data dataframe
			self.dataDict[timeFrameName]['data']['x'] = pd.DataFrame(x, columns = ['x'])
			self.dataDict[timeFrameName]['data']['y'] = pd.DataFrame(y, columns = ['y'])


	def getTopo(self):
		"""
		Reads topo file and returns dataframe with point cloud.

		-	Only supports GeoClaw topotype 2.
		"""

		zTopo = []

		#	Open file, remove blank lines
		with open(self.pathTopo) as myFile:
			topoDataLines = removeBlankLines(myFile)

			#	Read lines, look for data
			for num, line in enumerate(topoDataLines, 1):
				if 'ncols' in line:
					ncols = int(line[:23])
				elif 'nrows' in line:
					nrows = int(line[:23])
				elif 'xlower' in line:
					xlower = float(line[:23])
				elif 'ylower' in line:
					ylower = float(line[:23])
				elif 'cellsize' in line:
					cellsize = float(line[:23])
				elif 'nodata_value' in line:
					lineNoData = num
					noData = int(line[:23])
				elif num > lineNoData:
					zTopo.append(float(line[:23]))

		#	Create lists of x and y values
		xValues = list(np.arange(xlower, xlower + ncols * cellsize, cellsize))
		yValues = list(np.arange(ylower, ylower + nrows * cellsize, cellsize))

		#	Format x and y values to match fortran output
		x = xValues * nrows
		ygrid = []
		for yValue in yValues: ygrid.append([yValue] * ncols)
		y = [item for sublist in ygrid for item in sublist]

		#	Transpose to dataframe format
		metadata = list(map(list, zip(*[[ncols], [nrows], [xlower], [ylower], [cellsize], [noData]])))
		topoData = list(map(list, zip(*[x,y,zTopo])))

		#	Create pandas dataframes
		metadata = pd.DataFrame(metadata, columns = ['ncols', 'nrows', 'xlower', 'ylower', 'cellsize', 'noData'])
		topoData = pd.DataFrame(topoData, columns = ['x','y', 'zTopo'])

		return {'metadata':metadata, 'topoData':topoData}


	def addTopoToPointCloud(self, pointCloud):
		"""
		Adds topography to pointcloud.

		-	Done just in time to speed things up.
		"""

		#	Initialize topo data
		xTopo = np.array(self.topo['topoData']['x'])
		yTopo = np.array(self.topo['topoData']['y'])
		_zTopo = np.array(self.topo['topoData']['zTopo'])

		#	For each row in pointcloud, find matching topo value with mask
		for index, row in pointCloud.iterrows():
			mask = (xTopo == row['x']) & (yTopo == row['y'])
			row['zTopo'] = np.compress(mask, _zTopo)

		return pointCloud['zTopo']


	def plotSurf(self, xMin, xMax, yMin, yMax, zMin, zMax, camElev, camAngle):
		"""
		Plots surface for all time frames
		"""

		print('Started plotting surfaces...')

		#	Iterate through time frames
		for timeFrameName in self.fortQFileList:

			#	Trim spatial extent of point cloud with mask
			x = np.array(self.dataDict[timeFrameName]['data']['x'])
			y = np.array(self.dataDict[timeFrameName]['data']['y'])
			q1 = np.array(self.dataDict[timeFrameName]['data']['q(1)'])
			mask = ((x > xMin) & (x < xMax)) & ((y > yMin) & (y < yMax))
			x = np.compress(mask, x)
			y = np.compress(mask, y)
			q1 = np.compress(mask, q1)

			#	Add topography to pointcloud
			zTopo = self.addTopoToPointCloud(pd.DataFrame({'x': x, 'y': y, 'zTopo': np.nan}))

			print('Plotting ' + timeFrameName)

			#	Create surface of point cloud with 3d triangulation
			triang = tri.Triangulation(x, y)

			#	Trim non-real edges of the triangulation
			xScale = xMax - xMin
			yScale = yMax - yMin
			yPlotMin = yMin + 0.15 * yScale
			yPlotMax = yMin + 0.85 * yScale
			isbad = np.logical_or(np.greater(y,yPlotMax),np.less(y,yPlotMin))
			triang = tri.Triangulation(x, y)
			mask = np.all(np.where(isbad[triang.triangles], True, False), axis=1)
			triang.set_mask(mask)

			#	Plot surface
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			ax.plot_trisurf(triang, q1 + zTopo, linewidth=0.2, antialiased=True, edgecolor='none')
			
			#	Axis limits
			ax.set_xlim3d(xMin, xMax)
			ax.set_ylim3d(yPlotMin, yPlotMax)
			ax.set_zlim3d(zMin, zMax)
			#	Camera views
			ax.view_init(elev = camElev, azim = camAngle)
			#	Labels, title
			ax.set_title('Surface Plot of Tsunami')
			ax.set_xlabel('x (m)')
			ax.set_ylabel('y (m)')
			ax.set_zlabel('z (m)')
			
			# Save figures
			pathSurfPlots = self.pathOutput + 'surfPlots/'
			if os.path.exists(pathSurfPlots) == False:
				os.mkdir(pathSurfPlots)
			plt.savefig(pathSurfPlots+str(int(timeFrameName[-3:])+1)+'.jpg', format='jpg', dpi=1000, figsize=(5,8))
			plt.close()

		#	Create movie
		print('Done plotting surfaces. Creating movie...')
		os.system("ffmpeg -f image2 -framerate 12 -i {}%d.jpg {}movie.mp4".format(pathSurfPlots, self.pathOutput))


	def plotTransectY(self, xMin, xMax, yTransect, zMin, zMax, title):
		"""
		Plots transects for all time frames
		"""
		
		print('Started plotting transects...')

		#	Iterate through time frames
		for timeFrameIdx in range(len(self.fortQFileList)):

			timeFrameName = self.fortQFileList[timeFrameIdx]
			time = self.time[timeFrameIdx]
			# 	Trim spatial extent of point cloud
			x = np.array(self.dataDict[timeFrameName]['data']['x'])
			y = np.array(self.dataDict[timeFrameName]['data']['y'])
			q1 = np.array(self.dataDict[timeFrameName]['data']['q(1)'])
			mask = ((x > xMin) & (x < xMax)) & (y == yTransect)
			x = np.compress(mask, x)
			y = np.compress(mask, y)
			q1 = np.compress(mask, q1)

			#	Add topography to pointcloud
			zTopo = self.addTopoToPointCloud(pd.DataFrame({'x': x, 'y': y, 'zTopo': np.nan}))

			#	Store to dataframe and sort by x
			plotData = pd.DataFrame({'x': x, 'y': y, 'zTopo': zTopo, 'surf': q1 + zTopo}).sort_values(by=['x'])

			print('Plotting ' + timeFrameName)

			#	Plot surface
			fig = plt.figure()
			plt.plot( plotData['x'], plotData['surf'], 'b', plotData['x'], plotData['zTopo'], 'k')
			
			#	Axis limits
			plt.xlim((xMin,xMax))
			plt.ylim((zMin,zMax))

			#	Invert axes
			ax = plt.gca()
			ax.invert_xaxis()

			#	Labels, title
#			plt.legend('Topography','Water Surface')
			plt.title(title + " t = {} seconds".format(int(time)))
			plt.xlabel('x (m)')
			plt.ylabel('z (m)')
			
			#	Save figure
			pathTransectPlots = self.pathOutput + 'transectPlots/'
			if os.path.exists(pathTransectPlots) == False:
				os.mkdir(pathTransectPlots)
			plt.savefig(pathTransectPlots+timeFrameName+'Transect.jpg', format='jpg', dpi=1000, figsize=(5,8))
			plt.close()

		print('Done plotting transect.')


	def getTimeSeriesAtPoint(self, x, y):
		"""
		"""

		qTimeSeries = []

		#	Iterate through timeFrames
		for timeFrameIdx in range(len(self.fortQFileList)):

			#	Get data for given timeFrame
			timeFrameName = self.fortQFileList[timeFrameIdx]
			X = np.array(self.dataDict[timeFrameName]['data']['x'])
			Y = np.array(self.dataDict[timeFrameName]['data']['y'])
			Q = np.array(self.dataDict[timeFrameName]['data']['q(1)'])

			#	Find given x,y coordinate and corresponding q value
			mask = (X == x) & (Y == y)
			X = np.compress(mask, X)
			Y = np.compress(mask, Y)
			Q = np.compress(mask, Q)[0]
			qTimeSeries.append(Q)

		#	Add topography to pointcloud
		zTopo = self.addTopoToPointCloud(pd.DataFrame({'x': X, 'y': Y, 'zTopo': np.nan})).iloc[0]

		return qTimeSeries + zTopo


def removeBlankLines(textFile):
	allLines = (line.rstrip() for line in textFile)
	return (line for line in allLines if line)


def plotTimeSeriesAtPoint(inputs, x, y, pathOutput, legend, title=None, saveFig=False, figName=None, show=False):
	"""
	"""
	print('Plotting time series at x = {}, y = {}'.format(x,y))

	#	Create figure
	fig = plt.figure()
	plt.clf()

	#	Iteratively add plots
	for inputPair in inputs:
		if len(inputPair) == 2: plt.plot(inputPair[0], inputPair[1])
		elif len(inputPair) ==3: plt.plot(inputPair[0], inputPair[1], color=inputPair[2])
		elif len(inputPair) ==4: plt.plot(inputPair[0], inputPair[1], color=inputPair[2], linestyle=inputPair[3])

	#	Add labels, title
	plt.xlabel('Time (s)')
	plt.ylabel('Water Surface Elevation (m)')
	if title is None:
		title = "Time Series of Water Surface Elevation at x = {}, y = {}".format(x,y)
	plt.title(title)

	plt.legend(legend)
#	plt.legend(["Slope = 1/50 = 0.02", "Slope = 3/40 = 0.075"])

	#	Save or show
	if saveFig is True:
		if figName is None:
			figName = "x_{}_y_{}".format(x,y)
		pathTimeSeriesPlots = pathOutput + 'timeSeriesPlots/'
		if os.path.exists(pathTimeSeriesPlots) == False:
			os.mkdir(pathTimeSeriesPlots)
		plt.savefig(pathTimeSeriesPlots + figName + 'TimeSeries.jpg', format='jpg', dpi=1000, figsize=(5,8))
	if show is True:
		plt.show()
	print('Done plotting time series.')


def plotMultipleTransectsY(geoDataObjLst, xMin, xMax, yTransect, zMin, zMax, \
	pathOutput, legend, title=None, saveFig=False, figName=None, show=False, \
	plotType = 1):
	"""
	Similar to plotTransectY but does multiple lines.

	Assumes .fortq files in all cases "line up" and topo files are identical
	"""

	print('Started plotting transects...')

	#	Iterate through time frames
	for timeFrameIdx in range(len(geoDataObjLst[0].time)):

		#	Create blank figure
		fig = plt.figure()
		plt.clf()

		#	Do everything in one subplot
		if plotType == 1:
			#	-	Axis limits
			plt.xlim((xMin,xMax))
			plt.ylim((zMin,zMax))
			#	-	Invert axes
			ax = plt.gca()
			ax.invert_xaxis()
			#	-	Labels, title
			plt.title(title + " t = {} seconds".format(int(geoDataObjLst[0].time[timeFrameIdx])))
			plt.xlabel('x (m)')
			plt.ylabel('z (m)')

		if plotType == 2:
			nrows = len(geoDataObjLst)
			ncols = 1

		#	Iterate through geoData objects
		objCounter = 0
		for geoDataObj in geoDataObjLst:

			#	Get y-transect for given timeframe, spatial limits, and geoData object
			timeFrameName = geoDataObj.fortQFileList[timeFrameIdx]
			time = geoDataObj.time[timeFrameIdx]
			# 	Trim spatial extent of point cloud
			x = np.array(geoDataObj.dataDict[timeFrameName]['data']['x'])
			y = np.array(geoDataObj.dataDict[timeFrameName]['data']['y'])
			q1 = np.array(geoDataObj.dataDict[timeFrameName]['data']['q(1)'])
			mask = ((x > xMin) & (x < xMax)) & (y == yTransect)
			x = np.compress(mask, x)
			y = np.compress(mask, y)
			q1 = np.compress(mask, q1)

			#	Add topography to pointcloud
			zTopo = geoDataObj.addTopoToPointCloud(pd.DataFrame({'x': x, 'y': y, 'zTopo': np.nan}))
			#	Store to dataframe and sort by x
			plotData = pd.DataFrame({'x': x, 'y': y, 'zTopo': zTopo, 'surf': q1 + zTopo}).sort_values(by=['x'])

			print("Plotting {}".format(timeFrameName))

			#	Do things this way if everything should go on one set of axes
			if plotType == 1:
				if objCounter == 0: 
					plt.plot(plotData['x'], plotData['surf'], 'g-')
					xTopoPlot = copy(plotData['x'])
					zTopoPlot = copy(plotData['zTopo'])
				elif objCounter == 1:
					plt.plot( plotData['x'], plotData['surf'], 'c-')
				elif objCounter == 2:
					plt.plot( plotData['x'], plotData['surf'], 'b-')
				else: print("Error: code not designed for this.")

			#	Do it this way if we want multiple plots
			elif plotType == 2:

				#	-	Create subplot
				plt.subplot(nrows, ncols, objCounter+1, label=str(objCounter))
				#	-	Axis limits
				plt.xlim((xMin,xMax))
				plt.ylim((zMin,zMax))
				#	-	Invert axis
				ax = plt.gca()
				ax.invert_xaxis()
				#	-	Labels
				if objCounter == 0: plt.title("t = {} seconds".format(int(time)))
				if objCounter + 1 == len(geoDataObjLst): plt.xlabel('x (m)')
#				else: plt.gca().axes.get_xaxis().set_visible(False)
				ax.tick_params(axis="x", labelsize=7)
				plt.ylabel('z (m)')
				plt.plot(plotData['x'], plotData['surf'], 'b-', plotData['x'], plotData['zTopo'], "k")
				plt.text(x = xMin+0.175*(xMax-xMin), y = zMin+0.05*(zMax-zMin), s=legend[objCounter], fontsize=8)

			objCounter += 1

		#	Add topography at the end if all on one axis
		if plotType == 1:
			#	Plot topography
			plt.plot(xTopoPlot,zTopoPlot,"k")
			#	Legend
			plt.legend(legend)

		#	Save figure
		pathTransectPlots = pathOutput + 'transectPlots/'
		if os.path.exists(pathTransectPlots) == False:
			os.mkdir(pathTransectPlots)
		plt.savefig(pathTransectPlots+str(timeFrameIdx+1)+'.jpg', format='jpg', dpi=1000, figsize=(5,8))
		plt.close()

	#	Create movie
	print('Creating movie...')
	os.system("ffmpeg -f image2 -framerate 12 -i {}%d.jpg {}movie.mp4".format(pathTransectPlots, pathOutput))


if __name__ == '__main__':

	pass
