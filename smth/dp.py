class Deque:
    def __init__(self):
        self.arr = []

    def addFront(self, n):
        self.arr.insert(0, n)

    def addRear(self, n):
        self.arr.append(n)

    def removeRear(self):
        if not self.arr:
            print("Deque empty. Cannot remove from an empty deque.\n")
        else:
            self.arr.pop()

    def search(self, key):
        if not self.arr:
            print("Deque is empty. Cannot search.")
            return
        for i in range(len(self.arr)):
            if self.arr[i] == key:
                print("Found at index:", i)
                return
        print("Not found")

    def display(self):
        if not self.arr:
            print("Deque is empty.")
        else:
            print("Deque:", self.arr)


print("Aarjav Jain C14 2303063\n")

dq = Deque()

dq.addFront(100)
dq.addRear(200)
dq.addRear(300)
dq.addFront(50)

print("After adding elements:")
dq.display()
print("\n")

dq.removeRear()
print("After removing from rear:")
dq.display()
print("\n")

print("Searching for 100:")
dq.search(100)
print("\n")

print("Searching for 500:")
dq.search(500)
