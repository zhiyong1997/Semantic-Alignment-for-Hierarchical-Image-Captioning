import cv2, time
from utils_coco import show_img
from attention import Attention
import os, pickle

def rgbgrgbgr(img):
    temp = img.swapaxes(0, 2)
    temp = temp[::-1]
    img = temp.swapaxes(0, 2)
    return img

def load_img(img_path):    
        img = cv2.imread(img_path)
        img = rgbgrgbgr(img)
        return img

# ignore this now, refer to the main func
def show_sentence_attention(img, attention_array, cap, batch_index):
    # ==== folder to store the sentence attention ====
    folder = 'attentions/{}/'.format(batch_index)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # ==== write the cap ====
    with open(os.path.join(folder, 'cap.txt'), 'wb') as f:
        pickle.dump(cap, f)

    # ==== write origin image ====
    cv2.imwrite(os.path.join(folder, 'origin.jpg'), img)

    assert attention_array.shape[0] + attention_array.shape[1] == 20 + 196
    attention_array = attention_array.reshape((20, 14, 14))
    attention = Attention()

    # ==== for every word in the sentence ====
    # ==== draw the attention ====
    threshold = 1e-3
    for i in range(20):
        attention.build(img)
        for j in range(14):
            for k in range(14):
                if attention_array[i][j][k] > threshold:
                    attention.add_spot((j + 0.5) * 16, (k + 0.5) * 16, strength = attention_array[i][j][k])

        cv2.imwrite(folder + 'step {}.jpg'.format(i), attention.get_image())

if __name__ == '__main__':
    img_path = './figure/COCO_train2014_000000000025.jpg'
    img = load_img(img_path)
    attention = Attention()

    attention.build(img, 0.3)
    attention.add_spot(50, 50, strength = 0.5, dev = 25)
    attention.add_spot(150, 150, strength = 0.5)
    attention.show()
