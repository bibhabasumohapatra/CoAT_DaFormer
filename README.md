# CoAT with DAFormer #
*CoAT: Co-scale conv-attentional image Transformers
*DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation
### How to use ###
````
def Model(num_classes=4):
    encoder = coat_lite_medium()
    checkpoint = "downloads/coat_lite_medium_384x384_f9129688.pth"
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["model"]
    encoder.load_state_dict(state_dict, strict=False)
    net = Net(encoder=encoder, num_classes=num_classes).cuda()
    return net
	
my_model = Model(num_classes=6) 
# note: please check the path for checkpoint for encoder or install from pytorch-timm-models
````
