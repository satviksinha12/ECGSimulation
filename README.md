# ECGSimulation
This is an ECG program built using Tensorflow,Numpy and Keras.
Welcome to the ECGSimulation wiki!

This project was built as a part of College assignment and is meant for research purposes only. Not to be used in any sort of real life medical diagnosis prediction.

# Oreview into what this project is all About

Over decades ECG has been used to measure Electrical activity of Heart to diagnose Possible Heart attack or any irregular Heart beating. 
The aim of this project was to simulate The electrical activity and measure an irregularities in the wave form to predict possible issues on spot

![ECG view](https://www.bing.com/th?id=OIP.9oannh-Z3P8sI0vYy2h2YwAAAA&w=312&h=200&c=8&rs=1&qlt=90&o=6&dpr=1.4&pid=3.1&rm=2)

## Setting Things into motion

The initial plan was to capture real heart beat using sensors and Arduino and have tensorflow predict irregularities heartbeat but due to delay in logistics , I decided to do some extra digging and look up on how I can generate

This is where Numpy came to use

Having been through ECG myself when I ran into some health issues. I looked back at my old records to see how the pattern of ECG should be, Once that challenge was clear and planning was done we setted out to code in Visual Studio Code
![image](https://github.com/user-attachments/assets/edfb1e3b-00ee-4b40-b2b8-4a6449e1bf91)

We started off by Importing all the libraries that would be needed to Simulate ECG and carry out predictions on the graph in real time accordingly. A list of these libraries include

1-) Numpy-For Generating and simulating ECG signals.
2-) TensorFlow- For carrying out predictions
3-) Keras and it's Sequential model and dense,LSTM and Dropout layers (bundeled with tensorflow) 
4-) Adam optimiser(Included with keras in TensorFlow)

The first step was to declare simulation parameters , that was heartbeat, Simulation time in seconds and sampling frequency
![image](https://github.com/user-attachments/assets/43cc2347-6464-4308-968d-31888a670972)

The third step was to Generate a heartbeat 
For this there was a need to find a  BPM that can be used to generate a Waveforms on which TensorFlow can predict normal and abnormal electrical signals of heart 

For this 60 BPM was taken in account. When human body is at rest , the heartbeats can go down all the way to 60 BPM.When human body is in active state 60 BPM can mean bradycardia ,However in athletes it is normal for BPM's to touch low to 60

![image](https://github.com/user-attachments/assets/d3d7dcff-fa4f-4f25-8e8a-26323004a019)
 
In  ECG simulation there are 5 points in waveform, P Q R S T

![](https://glneurotech.com/FAQ/ecg_ekg_clip_image001.gif)

The P-Q segment represents the time the signals travel from the SA node to the AV node.  
The QRS complex marks the firing of the AV node and represents ventricular depolarization: 
Q wave corresponds to depolarization of the interventricular septum. 
R wave is produced by depolarization of the main mass of the ventricles. 
S wave represents the last phase of ventricular depolarization at the base of the heart.
Atrial repolarization also occurs during this time but the signal is obscured by the large QRS complex.
The S-T segment reflects the plateau in the myocardial action potential.  This is when the ventricles contract and pump blood. 
The T wave represents ventricular repolarization immediately before ventricular relaxation, or ventricular diastole. 
The cycle repeats itself with every heartbeat.

The same was generated using mathematical formulae of Numpy library, The most common Trigonometric Function Sine was used to generate Waveforms.
and the Generate ECG signal function was called 

![image](https://github.com/user-attachments/assets/65728b9f-f715-4bad-b4e6-a4457458fa6b)

To make graph look more realistic we added some noise to our waveforms


