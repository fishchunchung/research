import csv
import cv2
import numpy as np
import os



def save_csv(data,name, path):
    filename = name+".csv"
    text = open(os.path.join(path,filename), "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id",str(name)])
    for i in range(len(data)):
        s.writerow([i,data[i]]) 
    text.close()



def save_image(dir_name, name, data):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) , str(dir_name))
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path,str(name))
    cv2.imwrite(name,data)
    
if __name__ == "__main__":
    ans = []
    for i in range(10):
        ans.append(i-10)
    
    print(ans)
    
    save_csv(ans,"reward", "/home/chung/Desktop/Atari/reverse_parking/model")
    a = np.array(ans)
    save_image("model", "tttt.jpg", a)