# Practica-CUDA
Desarrollo de una versión paralelizada con [CUDA](https://developer.nvidia.com/cuda-zone) del programa propuesto.

CUDA es una plataforma de cálculo paralelo y un modelo de programación desarrollado por NVIDIA para el cálculo general en unidades de procesamiento gráfico (GPU). Con CUDA, los desarrolladores pueden acelerar drásticamente las aplicaciones de cálculo aprovechando la potencia de las GPU.

**Para ejecutar** es necesario tener una tarjeta gráfica de Nvidia y asegurarse durante el proceso de instalación de incluir el driver de **CUDA 11.2**, que substituye al que tenga el sistema para gestionar la tarjeta gráfica.

### Compilar un programa CUDA
```c
nvcc codigo.cu -o codigo -lcuda -arch=sm_61
```

### Ejecutar un programa CUDA
```c
./codigo
```
