#############################################SAVE DICT##############################################
#with open("dict.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict,fp)
#fp.close()
#############################################SAVE IMAGE LIST##############################################
#with open("list_img.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(image_list,fp)
#fp.close()
#############################################SAVE DICT_KEYS##############################################
#with open("dict_keys.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict_keys,fp)
#fp.close()

#with open("exif_lbl.txt", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(exif_lbl,fp)
#fp.close()


with open("dict.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	dict = pickle.load(fp)
fp.close()

with open("list_img.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	image_list = pickle.load(fp)
fp.close()

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

list1,list2 = get_np_arrays('cropped_arrays.npy')

class ConsistencyNet(tf.keras.Model):
  def __init__(self, siamese):
    super(ConsistencyNet, self).__init__()
    
    self.siamese = siamese
    for layer in self.siamese.layers:
      layer.trainable = False

    self.model= tf.keras.Sequential(
        [
          Dense(512, activation='relu'),  
          Dense(1, activation='sigmoid')
        ]
    )


  def call(self, inputs):   
    netInput = self.siamese(inputs)
    x = self.model(netInput)
    return x
#--------------------------------------------------------------------------------------------
EPOCHS = 100 
def datagenerator(images,images2, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            x = images[start:end] 
            #y = labels[start:end]
            x2 = images2[start:end]
            yield (x,x2)#,y

            start += batchsize
            end += batchsize

"""
print("[INFO] starting test")

#--------------------------------------------------------------- EXTRACT 
dict,image_list,dict_keys = extract_exif()
#--------------------------------------------------------------- REMOVE ELEMENTS
dict = remove_elements(dict)

print("[INFO] number of keys: ", len(dict_keys))
#--------------------------------------------------------------- CREATE SAMPLES
list1_img,list2_img = create_batch_samples(dict,image_list)
#--------------------------------------------------------------- GENERATE LABELS
exif_lbl = generate_label(dict_keys,list1_img,list2_img)

with open("exif_lbl.txt", "wb") as fp:   #Picklingpickle.dump(l, fp)#
	pickle.dump(exif_lbl,fp)
fp.close()
#--------------------------------------------------------------- CROP IMAGES
list1,list2 = cropping_list(list1_img,list2_img)


#--------------------------------------------------------------- GET ELEMENTS
#list1,list2 = get_np_arrays('test_cropped_arrays.npy')
#with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
#	exif_lbl = pickle.load(fp)
#fp.close()
#--------------------------------------------------------------- RUN MODEL
x_train = datagenerator(list1,list2,exif_lbl,32)
"""
path = r"/content/drive/MyDrive/foto/test/images"
dir = os.listdir(path)
print(dir)
"""
length = len(dir)
tmp1 = np.empty((length*32, 128, 128, 3), dtype=np.uint8)
tmp2 = np.empty((length*32, 128, 128, 3), dtype=np.uint8)
dir_counter = 0
internal_loop = 0
for elem in dir:
    print(dir_counter)
    elem = os.path.join(path,elem)
    foto1 = cv2.imread(elem)[:,:,[2,1,0]]
    while internal_loop<32:
        patch1 = util.random_crop(foto1,[128,128])
        patch2 = util.random_crop(foto1,[128,128])
        tmp1[dir_counter*32+internal_loop] = patch1
        tmp2[dir_counter*32+internal_loop] = patch2
        internal_loop +=1

    dir_counter +=1
    internal_loop = 0
x_train = datagenerator(tmp1,tmp2,32)
model = tf.keras.models.load_model('siameseMLP.h5')

model.evaluate(x_train,batch_size = 32)

print(model.metrics_names)
"""



