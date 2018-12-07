# The-Eye-in-the-Sky
Inter IIT Project

## Resources
- Article: [Learn How to Train U-Net On Your Dataset](https://medium.com/coinmonks/learn-how-to-train-u-net-on-your-dataset-8e3f89fbd623)
- Repo: [Previously implemented repo](https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation)
- Paper: [CNNs for large scale-remote sensing image Classification](https://arxiv.org/pdf/1703.00121.pdf)
- Paper: [RNNs to correct Satellite Image Classification Maps](https://arxiv.org/pdf/1608.03440.pdf)
- Article: [Deconvolution](https://distill.pub/2016/deconv-checkerboard/)
- Report: [Various indices from Satellite images](http://www.slopeproject.eu/public/deliverables/D201.pdf) 

### Color Codes for classes
- Water = [0,0,150]
- Trees = [0,125,0]
- Grass = [0,255,0]
- Trains = [255,255,0]
- Soil = [150,80,0]
- Roads = [0,0,0]
- Unlabelled = [255,255,255]
- Buildings = [100,100,100]
- Pools = [150,150,255]

### Pixelwise Percentage and Threshold values:
 - road:15.2          0.5
 - building:21.88     0.5
 - tree:8.5           0.4  
 - grass:6.88         0.4
 - soil:0.93          0.35
 - water:5.72         0.4
 - rail:1.0           0.35
 - pool:0.15          0.3
 - unlabelled:39.72   0.6
