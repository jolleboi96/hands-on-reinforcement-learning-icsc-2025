# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:44:50 2021

@author: Joel Wulff
"""

# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

#SAVE_PATH = r"H:\user\j\jwulff\SWAN_projects\batch_jobs2\outputs\datamatrix_files\quad\assembled_lookup_tables\quad_datamatrix"
#DATA_PATH = r"H:\user\j\jwulff\SWAN_projects\batch_jobs2\outputs\datamatrix_files\quad\machine_settings_fixed"
DATA_PATH = "./double/profiles" # Generalized path



first_file = True


class Datamatrix_lookup_class_double():
    
    '''
    Class that searches through the provided data path, reading filenames and adding datasamples
    to a dictionary that can then be queried using the key dict[p42]. Comes with functions to get
    or interpolate samples in between simulated samples.
    '''

    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.filepaths = []
        self.dictionary = None # key 1 is p42*100, key 2 is p84 * 100, value is filepath
        self.p42_labels = None
        #self.p84_labels = None
        first_file=True
        for file in os.listdir(self.data_path):
        #print(file)
            if file.endswith("profile.npy"):
                # print path name of selected files
                file_path = os.path.join(self.data_path, file)
                self.filepaths.append(file_path)


                #print(file_path)

                # Extract phases
                split = file.split('_')
                p42 = int(split[1])
                #p84 = int(split[3])
                if first_file:
                    phases_filepaths = np.array([p42, file_path], dtype=object)# phase 42, phase 84, datamatrix)
                    first_file = False
                else:    
                    #loaded = np.load(file_path, allow_pickle=True)
                    phases_filepaths = np.vstack((phases_filepaths, np.array([p42, file_path], dtype=object)))
        
# loaded_values now contain all information needed to construct the dictionary
# Format: [[phase_42, phase_84, objective_function values], ... ]

        nbr_samples, cols = np.shape(phases_filepaths)
        uniq_p42 = sorted(set(phases_filepaths[:,0]))
        


        #self.p84_dictionary = {}
        self.dictionary = {} # row = p42, value is filepath to datamatrix
        self.step_size = uniq_p42[1]-uniq_p42[0]
        self.p42_labels = uniq_p42
       #self.p84_labels = uniq_p84
        
        #row_dict = {}
        for i, p1 in enumerate(uniq_p42):
            col_dict = {}
            #self.p42_labels[i] = p1
            p1_idx = np.where(phases_filepaths[:,0]==p1)
            filepath =  phases_filepaths[p1_idx,1][0][0]
            self.dictionary["{}".format(p1)] =  filepath
            
        print('Data dictionary compiled.')


    

    def get_profile(self, p42, norm=True):
        p42 = int(p42*100)
        path = self.dictionary["{}".format(p42)]
        dir_name = os.path.dirname(path) + r'\profiles'
        file = os.path.basename(path)
        split = file.split('_')
        split[-1] = 'profile.npy'
        new_name = '_'.join(split)
        new_path = dir_name + r'\{}'.format(new_name)
        profile = np.load(new_path)
        if norm:
            profile[1] = profile[1]/np.max(profile[1])
        return profile[1] # For now only return values, not time

    def get_interpolated_profile(self, x, norm=True):  # Insert any continuous p42,p84 and get an interpolated matrix
        # Find closest value of x and y in p42/p84 labels
        x = int(x*100)
        abs_diff_function1 = lambda list_value: abs(list_value - x)
        closest_p42 = min(self.p42_labels, key=abs_diff_function1)
        if x>=4500:
            x1 = closest_p42
            x0 = closest_p42
        elif x<=-4500:
            x1 = closest_p42
            x0 = closest_p42
        elif closest_p42 > x:
            x1 = closest_p42
            x0 = (closest_p42-self.step_size)
        elif closest_p42 <= x:
            x1 = (closest_p42+self.step_size)
            x0 = closest_p42

        if x1==x0:
            xd =0
        else:
            xd = (x/100-x0/100)/(x1/100-x0/100)

        y1 = np.load(self.dictionary[str(x1)])
        y0 = np.load(self.dictionary[str(x0)])

        if norm:
            y1 = y1[1]/np.max(y1[1])
            y0 = y0[1]/np.max(y0[1])


        
        # Bilinear interpolation

        # x-direction

        y = y0 + (xd)*(y1-y0)

        return y


    def get_datamatrix(self, p42, p84):
        p42, p84 = p42*100, p84*100
        path = self.dictionary["{}".format(p42)]["{}".format(p84)]
        return np.load(path)

    def get_interpolated_matrix(self, x):  # Insert any continuous p42,p84 and get an interpolated matrix
        # Find closest value of x and y in p42/p84 labels
        x = int(x*100)
        abs_diff_function1 = lambda list_value: abs(list_value - x)
        closest_p42 = min(self.p42_labels, key=abs_diff_function1)

        if x>=4500:
            x1 = closest_p42
            x0 = closest_p42
        elif x<=-4500:
            x1 = closest_p42
            x0 = closest_p42
        elif closest_p42 > x:
            x1 = closest_p42
            x0 = closest_p42-self.step_size
        elif closest_p42 <= x:
            x1 = closest_p42+self.step_size
            x0 = closest_p42
    

        # Compute distances
        if x1==x0:
            xd =0
        else:
            xd = (x/100-x0/100)/(x1/100-x0/100)

        y1 = np.load(convert_profile_path_to_dm(self.dictionary[str(x1)]))
        y0 = np.load(convert_profile_path_to_dm(self.dictionary[str(x0)]))

        y = y0 + (xd)*(y1-y0)

        return y
        
    def get_available_p42(self):
        return self.p42_labels

    def get_available_p84(self):
        return self.p84_labels

def convert_profile_path_to_dm(path):
    dir_name = os.path.dirname(os.path.dirname(path))
    file = os.path.basename(path)
    split = file.split('_')
    split[-1] = 'datamatrix.npy'
    new_name = '_'.join(split)
    new_path = dir_name + r'\{}'.format(new_name)
    return new_path

def compute_2d_distance(x1,y1, x2,y2):
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def deg2idx(p42,p84, lookup_class):
     
    p42max = max(lookup_class.p42_labels)
    p42min = min(lookup_class.p42_labels)
    p84max = max(lookup_class.p84_labels)
    p84min = min(lookup_class.p84_labels)
    nbr_of_p42 = len(lookup_class.p42_labels)
    nbr_of_p84 = len(lookup_class.p84_labels)
    idx = [p42+10,p84+10]
    return idx

def compute_2d_distance(x1,y1, x2,y2):
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist




# %%
# make plot
# fig, ax = plt.subplots()
  
# # show image
# #shw = ax.imshow(matrix[:,:,3], extent=[min(uniq_p84),max(uniq_p84), max(uniq_p42), min(uniq_p42)])
  
# # make bar
# bar = plt.colorbar(shw)
  
# # show plot with labels
# plt.xlabel('phi_84')
# plt.ylabel('phi_42')
# bar.set_label('ColorBar')
# plt.show()

if __name__ == "__main__":
    Data = Datamatrix_lookup_class_double()
    testing = Data.get_interpolated_profile(0.25)

    #Test interpolation

    interpol1 = Data.get_interpolated_matrix(45)
    interpol2 = Data.get_interpolated_profile(-1.5)
    interpol3 = Data.get_interpolated_profile(0)
    interpol4 = Data.get_interpolated_profile(1.5)
    test = convert_profile_path_to_dm(Data.dictionary['2000'])
    plt.figure()
    plt.subplot(141)
    plt.plot(interpol1)
    plt.subplot(142)
    plt.plot(interpol2)
    plt.subplot(143)
    plt.plot(interpol3)
    plt.subplot(144)
    plt.plot(interpol4)

    plot_phases = np.array(p42_cand[20:62][::5])
    fig, axs = plt.subplots(9,9, sharex=True, sharey=True)
    for i, p42 in enumerate(plot_phases):
        for j, p84 in enumerate(plot_phases):
            data = Data.get_datamatrix(int(p42/100),int(p84/100))
            axs[i,j].title.set_text("{}_{}".format(int(p42/100),int(p84/100)))
            axs[i,j].imshow(data)
    plt.show()
            
    matrix = Data.get_datamatrix(0,0)
    plt.imshow(matrix)
    print("hej")