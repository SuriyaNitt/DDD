def read_train_targets():
    driverId = []
    trainId = []
    trainTarget = []
    # Check the existence of data directory
    if not os.path.isdir('../input/Training_Dataset'):
        print('Training Dataset is not found in path\n \
               Make sure that it is found in ../input/Training_Dataset folder')
    else:
        path =  '../input/Training_Dataset'
        driverFolders = os.listdir(path)
        driverFolders = sorted(driverFolders, key = natural_keys)
        for driverFolder in driverFolders:
            glassFolders = os.listdir(os.path.join(path, driverFolder))
            glassFolders = sorted(glassFolders, key = natural_keys)
            for glassFolder in glassFolders: 
                videos = glob.glob(os.path.join(path, driverFolder, glassFolder, '*.avi'))
                for video in videos:
                    fileBase = os.path.basename(video)
                    driverIdLocal, trainIdLocal, trainTargetLocal = populate_train_info(path, driverFolder, \
glassFolder, video)
def read_validation_targets():
