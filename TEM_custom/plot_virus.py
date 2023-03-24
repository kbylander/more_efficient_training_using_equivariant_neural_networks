name_classes = ['Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza', 'Lassa', 'Marburg', 'Nipah', 'Norovirus', 'Orf', 'Papilloma', 'Rift Valley', 'Rotavirus']

data = np.loadtxt('/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_noaug_93samplesperclass_14classes.txt',delimiter=' ')
images=data[:, :-1].reshape(-1,256,256).astype(np.float32)
labels=data[:,-1].astype(np.int64)
num_labels=len(labels)

unique_virus={}
for i,img in enumerate(images):
    if unique_virus.get(labels[i])==None:
        unique_virus[labels[i]]=img

from matplotlib import pyplot as plt
figure,axis=plt.subplots(4,4)
figure.tight_layout(h_pad=1.0,w_pad=0.2)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.5)
 
#plt.show()

#plt.autoscale(enable=True,tight=True)

axis_index=[axis[0,0],axis[0,1],axis[0,2],axis[0,3],axis[1,0],axis[1,1],axis[1,2],axis[1,3],axis[2,0],axis[2,1],axis[2,2],axis[2,3],axis[3,0],axis[3,1],axis[3,2],axis[3,3]]

for (ind,img) in unique_virus.items():
    axis_index[ind].imshow(im, interpolation='nearest')
    axis_index[ind].set_title(name_classes[ind])
    axis_index[ind].set_yticklabels([])
    axis_index[ind].set_xticklabels([])
    axis_index[ind].set_xticks([])
    axis_index[ind].set_yticks([])

axis_index[-1].set_axis_off()

axis_index[-2].set_axis_off()

plt.show()
