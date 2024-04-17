# Common cluster explanations (Deepcover and Traditional approaches - GradCam etc.)
## Modifications

Original deepcover code is modified to support other explainability methods. GradCam is supported now, other methods will be added shortly. This provides common explanations pulled from multiple sources of explanations.

# Collection of Explainablity methods, including DeepCover and GradCam
```
python ./src/deepcover.py --help
usage: deepcover.py [-h] [--model MODEL] [--inputs DIR] [--outputs DIR]
                    [--measures  [...]] [--measure MEASURE] [--mnist-dataset]
                    [--normalized-input] [--cifar10-dataset] [--grayscale]
                    [--vgg16-model] [--inception-v3-model] [--xception-model]
                    [--mobilenet-model] [--attack] [--text-only]
                    [--input-rows INT] [--input-cols INT]
                    [--input-channels INT] [--x-verbosity INT]
                    [--top-classes INT] [--adversarial-ub FLOAT]
                    [--adversarial-lb FLOAT] [--masking-value INT]
                    [--testgen-factor FLOAT] [--testgen-size INT]
                    [--testgen-iterations INT] [--causal] [--wsol FILE]
                    [--occlusion FILE]
                    [--explainable-method STR] 
```

## To run GradCam based explanation on your pre-trained model
```
python ./src/deepcover.py --model [YOUR_MODEL] --inputs [YOUR_DATA] --outputs [YOUR_OUTPUT_DIR] --explainable-method GradCam
```

## To run the DeepCover based explanation on your pre-trained model:
```
python ./src/deepcover.py --model [YOUR_MODEL] --inputs [YOUR_DATA] --outputs [YOUR_OUTPUT_DIR] --explainable-method DeepCover
```

## Other options
```
python src/deepcover.py --mobilenet-model --inputs data/panda/ --outputs outs --measures tarantula zoltar --x-verbosity 1 --masking-value 0
```
`--measures`      to specify the SFL measures for explaining: tarantula, zoltar, ochiai, wong-ii

`--x-verbosity`   to control the verbosity level of the explanation results

`--masking-value` to control the masking color for mutating the input image


## To start running the causal theory based explaining:
```
python ./sfl-src/sfl.py --mobilenet-model --inputs data/panda --outputs outs --causal --testgen-iterations 50
```
`--causal`              to trigger the causal explanation

`--testgen-iterations`  number of individual causal refinement calls; by default, it’s 1  

## To load your own model
```
python src/deepcover.py --model models/gtsrb_backdoor.h5 --input-rows 32 --input-cols 32 --inputs data/gtsrb/ --outputs outs
```
`--input-rows`    row number for the input image

`--input-cols`    column number for the input image
