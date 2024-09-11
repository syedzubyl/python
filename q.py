myQueue=list()
queuesize=10
def isEmpty(myQueue):
    if len(myQueue)==0:
        return True
    else:
        return False
def dequeue(myQueue):
    if not(isEmpty(myQueue)):
        return myQueue.pop(0)
    else:
        return "queue is empty"
def size(myQueue):
    return len(myQueue)
while (True):
  print("queue operation \n")
  print("1.Enqueue operation \n")
  print("2.Dequeue operation \n")
  print("3.size of queue \n")
  print("4.Display queue \n")
  print("5.Exit \n")
  choice =int(input("enter your choice:"))
  if choice==1:
    if len(myQueue)==queuesize:
        print("queue is full")
    else:
         n=int(input("enter no.of.elements for enqueue operation:"))
    if n<=queuesize:
        for i in range(0,n):
            element=input("enter the number to add item to queue:")
            myQueue.append(element)
            print("no.of.items added to queue \n",i+1)
            continue
  elif choice==2:
   qitem=dequeue(myQueue)
   print("item removed is:",qitem)
   continue
  elif choice==3:
   print("no.of.items in queue=",size(myQueue))
   continue
  elif choice==4:
   print(myQueue)
  elif choice==5:
   break 
else:
  print("invalid choice")
         