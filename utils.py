import numpy as np

def read_xyz(filename):
    atm_names = []
    coords = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue  
            
            atm_names.append(parts[0])  
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords, dtype=float)
    return atm_names, coords