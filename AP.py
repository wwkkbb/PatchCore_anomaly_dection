path=r'/home/burly/bigimg.txt'
file = open(path, 'r')
s=file.readlines()
pixel_auc,img_auc,AP=0,0,0
for i in range(len(s)):
    class_=eval(s[i])
    for key in class_:
        x=class_[key]
        pixel_auc+=x["pixel_auc"]
        img_auc+=x["img_auc"]
        AP+=x["AP"]
mean_pixel=pixel_auc/10
mean_img_auc=img_auc/10
mean_AP=AP/10
# 关闭文件
file.close()
file = open(path, 'a')
file.write(str( {'mean_pixel': mean_pixel, 'mean_img_auc': mean_img_auc, 'mean_AP': mean_AP}))
file.close()