import fasttext
for ep in [7, 12, 25, 30, 50, 70, 100]:
    model = fasttext.train_supervised(input="./abstract_cate.train", dim=128,\
             loss='softmax', thread=128,  minn=1, maxn=5, wordNgrams=4, epoch=ep)
    
    result = model.test("abstract_cate.test")
    print("epoch:{}".format(ep), result)
    model.save_model("./abstract_cate_ep{}.bin".format(ep))
