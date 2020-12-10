# ABSTRACT
Safety and security has become a concern on every level whether it is on individual level or group of individuals. To visualize and remain conscious about the surroundings in order to know what’s happening has become crucial and essential especially where visual sight is unable to detect the objects like in darker areas. Therefore, we need to develop the mobile app which is capable to detect the surrounding environment in a thermal mode and let the user know in real time about the entities present in his/her surroundings. The hardware to be used is the thermal mobile camera attached with the android phone which will take the input feed in real time and after processing the data, it will display the required results accordingly. The idea here is to train deep learning model first and then run its inference on mobile app because due to this inference the model is capable enough to run on low end powered devices. For model to run on low end powered devices MobileNet architecture is used due to its lightweight and fast neural architecture which reduce the number of parameters drastically. The model uses depthwise separable convolution approach to reduce and optimize the model. The depthwise separable convolution is made up of two layers, firstly it uses depthwise convolution in which single filter is applied to each input channel and then when it comes to pointwise convolution it uses 1 x 1 convolution, after which all the layers are combined to give the output result. The resulted optimized file upon deployment in android will be used to classify the objects without any need of wireless connectivity and reducing the latency. The dataset collected was on three different categories including human, cat and car using the Flir thermal camera and Seek thermal camera. Supervised learning model was trained with using seek thermal camera and tested with images which were taken by Flir camera and the testing accuracy was 90 percent on them.



# Table of Contents
	
1.1	Introduction	

1.2	Motivation	

1.3	Scope	

1.4	Structure	

2.1	Thermal Imaging Camera	

2.2	How Does Thermal Camera Works	

2.3	How Its Different Than Night Vision	

2.4	Applications of Thermal Imaging	

3.1	System level Diagram 

3.2	Software	

    Knowledge Base 

3.3	Data Collection	

3.4 Mobilenet Architecture 

3.5	Flow Diagram 

3.6	Software Tools 
	
    Tenosrflow Lite Framework 
	
    Android Studio 
	
    Flir SDK 
	
    Mobile App Screen 

3.7	Hardware	

3.8 Seek Thermal and Flir Specs 

4.1 Android App 

5.1 Performance and Tuning Parameters 

5.2	Accuracy and Evaluation	

6.1	Conclusion	

6.2	Future Prospects	

References 




# Chapter 1: Introduction 

# 1.1	Introduction
Human eyes can only be able to detect that can only be able to detect electromagnetic radiations in the range of visible spectrum while other ranges such as infrared are not visible by our eyes.
The discovery of the infrared goes back in 1800 by Sir Frederick William Herschel. To see thermal difference between various colors of light, he managed to get sunlight passed through prism glass which created spectrum and consequently observed the temperature of each color. While noticing the temperatures of the colors there was increase in temperature from violet to red.
Now having the temperature pattern across different colors, he wanted to extend his research beyond the colors especially red color. He concluded that the temperature was higher in this region than the region before.
The range of infrared radiation is between microwave and visible part of the spectrum. The infrared radiation originates due to heat or thermal radiation. If the temperature of any object is 0 Kelvin or -273.15 Celsius, it will emit radiation in the range of infrared. 
 


Visualizing diminished and faded objects has always been a hurdle in human day to day life. It has always been desired that clear and unambiguous things can be seen even if it is dark and in faded light. It’s really important to be aware of surrounding so that one is able to know what’s happening nearby. Therefore, that said, anyone can be able to continue to resume the activities he/she are doing if he/she exactly classifies what objects are there in front of him/her.




