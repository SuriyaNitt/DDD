'''
    Read the param value from config
'''
def read_config(param):
    configFile = open('config.txt', 'r')
    value = 0
    while(1):
        line = configFile.readline()
        if(line == '')
            break
        arr = line.strip().split(',')
        if arr[0] == param:
            value = arr[1]
            break
    return value
