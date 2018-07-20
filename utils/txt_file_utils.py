## opening the text file 
f = open('ilsvrc_train_new.txt', 'r').readlines()
f1 = open('ilsvrc_train_n.txt', 'a')
## Changing the strings

for i in range(len(f)):
	temp_line = f[i]
	temp_line_replaced = temp_line.replace('krishna', 'krishna/tfff')
	f1.write(temp_line_replaced)
	if i%1000 == 0:
		print(i/1000)




