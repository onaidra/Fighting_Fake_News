"""
im1 = cv2.imread(r"C:\Users\Adri\Desktop\VISIOPE\prova\foto\D01_Motorola_E3_1\orig\D01_img_orig_0001.jpg")[:,:,[2,1,0]]
im1  = util.random_crop(im1,[128,128])
list1 = []
list2 = []
for i in range(10):
    list1.append(im1)
    list2.append(im1)
exif_lbl = np.ones((2,83))
exif_lbl[1] = np.random.randint(0,2,(1,83))
cls_lbl = np.ones((1,1))
cls_lbl[0][0] = 1
"""
"""
#------------------------------------------------------------------------------
im1 = [image_list[0]]
im2 = [image_list[1]]
exif_lbl = generate_label(im1,im2)
list1,list2 = cropping_list(im1,im2)
#-------------------------------------------------------------------------------


second_image_list = random_list(image_list)
exif_lbl = generate_label(image_list,second_image_list)
list1 = []
list2 = []
#tmp1 = np.empty((N, 128, 128, 3), dtype=np.uint8)
#tmp2 = np.empty((N, 128, 128, 3), dtype=np.uint8)
for i in range(10):
    print(second_image_list[i])
    x = cv2.imread(image_list[i])[:,:,[2,1,0]]
    y = cv2.imread(second_image_list[i])[:,:,[2,1,0]]
    patch1 = util.random_crop(x,[128,128])
    patch2 = util.random_crop(y,[128,128])

    list1.append(patch1)
    list2.append(patch2)

"""