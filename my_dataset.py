import random


class MyDataset:
    def __init__(self, all_datas, batch_size=1, shuffle=True):
        self.all_datas = all_datas
        # print(f"MyDataset self.all_datas:{self.all_datas}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.curser = 0

    def __iter__(self):
        print("hello __iter__")
        self.curser = 0
        if self.shuffle:
            random.shuffle(self.all_datas)
        return self

    def __next__(self):
        if self.curser >= len(self.all_datas):
            raise StopIteration
        batch_data = self.all_datas[self.curser: self.curser+self.batch_size]
        self.curser += self.batch_size
        return batch_data

    def __len__(self):
        return len(self.all_datas)

def main():
    list1 = [1, 2, 3, 4, 5, 6, 7]
    batch_size = 2
    shuffle = True
    epoch = 2
    mydataset = MyDataset(list1, batch_size, shuffle)

    for i in range(epoch):
        for batch_data in mydataset:
            print(batch_data)

if __name__ == "__main__":
    main()
