import numpy as np

def get_server_data(txt_path):
    datas = np.loadtxt(txt_path, dtype=str)
    new_datas = []
    for data in datas:
        new_data = "data/custom/images/" + data + ".jpg"
        new_datas.append(new_data)
        print(new_data)
        # print(data)
    np.savetxt("test.txt", np.array(new_datas), fmt="%s", delimiter=" ")

if __name__ == '__main__':
    get_server_data(txt_path="../txts/test.txt")