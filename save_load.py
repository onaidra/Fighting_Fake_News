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
