import os
import matplotlib.pyplot as plt

def plot_model_acc(model_list,model_acc_list,file_name, save_folder = 'result',fig_size = (12,12)):
    if os.path.isdir('./'+save_folder) == False:
        os.mkdir('./'+save_folder)
    else:
        print('already exist the folder in this path : {}'.format('./'+save_folder))
    
    f1 = plt.figure(figsize=fig_size)
    
    plt.xlabel('number of feature')
    plt.ylabel('acc')
    
    idx = 0
    for name,_ in model_list:
        plt.plot(range(len(model_acc_list[idx])), model_acc_list[idx], label=name)
        idx = idx+1
        
    plt.legend()
    f1.savefig(save_folder + '/'+file_name+'.png')

    f2 = plt.figure(figsize=fig_size)
    
    plt.xlabel('number of feature')
    plt.ylabel('acc')
    plt.ylim((0.0,1.0))
    
    idx = 0
    for name,_ in model_list:
        plt.plot(range(len(model_acc_list[idx])), model_acc_list[idx], label=name)
        idx = idx+1
        
    plt.legend()
    f2.savefig(save_folder + '/'+file_name+'_0to1'+'.png')
    
def plot_random_model_acc(model_list,model_mean_acc_list,
                          model_std_list,file_name ='random_output',save_folder = 'result',fig_size = (12,12)):
    
    if os.path.isdir('./'+save_folder) == False:
        os.mkdir('./'+save_folder)
    else:
        print('already exist the folder in this path : {}'.format('./'+save_folder))
        
    f1 = plt.figure(figsize=fig_size)
    plt.xlabel('number of feature')
    plt.ylabel('acc')
    
    idx = 0
    for name,_ in model_list:
        #print(name)
        plt.plot(range(len(model_mean_acc_list)), model_mean_acc_list[:,[idx]],label = name)        
        idx = idx+1
          
    plt.legend()
    f1.savefig(save_folder + '/'+file_name+'.png')
    
    f2 = plt.figure(figsize=fig_size)
    plt.xlabel('number of feature')
    plt.ylabel('acc')
    
    plt.ylim((0.0,1.0))
    idx = 0
    for name,_ in model_list:
        #print(name)
        plt.plot(range(len(model_mean_acc_list)), model_mean_acc_list[:,[idx]],label = name)
        idx = idx+1
        
        
    plt.legend()
    f2.savefig(save_folder + '/'+file_name+'_0to1'+'.png')
    
def plot_cluster_model_acc(model_list,model_acc_list,file_name ='cluster_output',
                           save_folder = 'result',fig_size = (12,12)):
    if os.path.isdir('./'+save_folder) == False:
        os.mkdir('./'+save_folder)
    else:
        print('already exist the folder in this path : {}'.format('./'+save_folder))
    
    f1 = plt.figure(figsize=fig_size)
    plt.xlabel('number of cluster')
    plt.ylabel('acc')
    
    idx = 0
    for name,_ in model_list:
        #print(name)
        plt.plot(range(len(model_acc_list)), model_acc_list[:,[idx]],label = name)
        idx = idx+1
        
        
    plt.legend()
    f1.savefig(save_folder +'/'+file_name+'.png')
    
    f2 = plt.figure(figsize=fig_size)
    plt.xlabel('number of cluster')
    plt.ylabel('acc')
    plt.ylim((0.0,1.0))
    
    idx = 0
    for name,_ in model_list:
        #print(name)
        plt.plot(range(len(model_acc_list)), model_acc_list[:,[idx]],label = name)
        idx = idx+1
        
        
    plt.legend()
    f2.savefig(save_folder +'/'+file_name+'_0to1'+'.png')