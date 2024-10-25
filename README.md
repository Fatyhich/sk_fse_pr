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

### Easy start

1. You need to clone this repo

```bash
git clone https://github.com/Fatyhich/sk_fse_pr.git
```

2. Then you need to build image with our project

```bash
cd sk_fse_pr
docker build . -t <your name of image>
```
If there are no errors in terminal, that means that all requirements for our project has been successfully downloaded and builded. Additionally, pretained weights for SAM has been downloaded. They stores in container in according folder.    

3. Starting container

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest
```

We separated the pipeline on 3 stages. Inside container you can rub them separately or by one-command.

Each stage description:

- Preprocessing Stage
We are doing sampling several images from hole dataset and store them in other folder named `preprocess_data/frames`  

To run this stage from local machine:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest make preprocess 
```

- Processing Stage
On this step we are generate masks for each picture from folder `preprocess_data/frames` using SAM. If masks was generated, they stored in new folder `preprocess_data/masks_area`  

To run this stage from local machine:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest make process 
```

- Postprocessing Stage
On the last step we applying generated masks on each image, according on to other. And as output we receive folder `output`, that consist of final images.    

To run this stage from local machine:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest make postprocess 
```

Also, you can run all this steps with one command from your local machine terminal:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest make all
```

If you want to remove processed data, here is command available:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest make remove
```

## Test

Despite the fact that tests are already checked and done on image's building stage, you can run them by the following commands:

```bash
python3 -m unittest -v test/test_<target_stage>.py
```

*Attention!* You need run them from inside of started container. If you want to run test from your local machine, without connecting to container:

```bash
docker run -it -v <place_of_dataset>:/mnt/data <your name of image>:latest python3 -m unittest -v test/test_<target_stage>.py
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


