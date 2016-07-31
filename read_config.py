'''
    Read the param value from config
'''
def read_config(param):
    configFile = open('config.csv', 'r')
    value = 0
    found = 0
    while(1):
        line = configFile.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        if arr[0] == param:
            value = arr[1]
            found = 1
            break
        
    if found == 0:
        print('No param called:{} found'.format(param))
    else:
        print('Param:{} found'.format(param))        
    return int(value)
