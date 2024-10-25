# Automated pipeline for processing images with SAM

**[MRob, Laboratory of Mobile Robotics, Skoltech  ](https://sites.skoltech.ru/mobilerobotics/)**

**Team #3 Project**

*Team members:*
- [ Maksim Alymov ](https://t.me/malyO2); 
- [ German Devchich ](https://t.me/gdevch); 
- [ Denis Fatykhoph ](https://t.me/didirium); 
- [ Maksim Smirnov ](https://t.me/msm1rnov)

The **key idea** of this project is to create an automated pipeline for processing images from a dataset, consisting of numerous pictures extracted from videos recorded in various outdoor and indoor environments. We utilize available odometry measurements to project human steps onto the ground. These projected points are then used as input for the Segment Anything Model (SAM). For more information on SAM, you can refer to the [ original repository ](https://github.com/facebookresearch/segment-anything/tree/main).
Below are provided examples of model results.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Getting Started 

1. To build conatiner:
```bash
docker build . -t fse_test
```
2. To run container:
```bash
docker run -it -v <place_of_dataset>:/mnt/data fse_test:latest 
```
3. To run container with prerecorded dataset:
```bash
docker run -it -v ./dataset:/mnt/data fse_test:latest
```
4. Download weights for SAM:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


