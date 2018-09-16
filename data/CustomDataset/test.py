
with open('trainval.txt', "r") as lf:
 for line in lf.readlines():
  print ((line),repr(line))
  img_file, anno = line.strip("\n").split(" ")
  print(repr(img_file), repr(anno))
