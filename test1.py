file1 = open('D:/毕业设计/实验/dataset/mot/train/MOT17-09-FRCNN/gt/gt.txt',"r")
a = file1.readlines()
file2 = open('D:/毕业设计/实验/dataset/mot/train/MOT17-04-DPM/gt/gt.txt',"r")
b = file2.readlines()

na = []
nb = []

for i in range(len(a)):
    if(a[i] != '\n'):
        na.append(a[i])
for i in range(len(b)):
    if(b[i] != '\n'):
         nb.append(b[i])
a = na
b = nb

same = True
if(len(a) != len(b)):
    same = False
else:
    for i in range(len(a)):
        if(a[i].strip() != b[i].strip()):
            same = False
            print("Falure is in line ",i," .")
if(same == False):
    print("Falure")
else:
    print("Success")
