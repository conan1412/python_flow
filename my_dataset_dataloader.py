import numpy as np

from my_dataset import MyDataset



class MyDataloader(MyDataset):
    def __init__(self, all_datas, batch_size=1, shuffle=True):
        '''如果不用super，就需要把父类所有的init里定义的变量都重写一遍，否则这些变量只会是默认值
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        # print(f"MyDataloader self.all_datas:{self.all_datas}")
        # print(f"MyDataloader self.batchsize:{self.batch_size}")
        '''
        super().__init__(all_datas, batch_size, shuffle)
        self.indexes = [i for i in range(len(self.all_datas))]


    def __iter__(self):
        # print("hello __iter__")
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.curser = 0
        return self

    def __next__(self):
        if self.curser >= len(self.all_datas):
            raise StopIteration
        # batch_data = self.all_datas[self.curser: self.curser+self.batch_size]
        batch_indexes = self.indexes[self.curser: self.curser+self.batch_size]
        self.curser += self.batch_size
        return self.all_datas[batch_indexes]



def main():
    list1 = np.array([1, 2, 3, 4, 5, 6, 7])
    batch_size = 2
    shuffle = True
    mydataloader = MyDataloader(list1, batch_size, shuffle)

    epoch = 2
    for i in range(epoch):
        for batch_data in mydataloader:
            print(batch_data)
    # print(mydataloader.all_datas)

if __name__ == '__main__':

    main()