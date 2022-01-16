# greek-parliament-proceedings-analysis

## Dataset
Greek Parliament Proceedings, 1989-2020 includes 1,280,918 speeches (rows) of Greek parliament members with a total volume of 2.30 GB, that were exported from 5,355 parliamentary sitting record files. They extend chronologically from early July 1989 up to late July 2020.  
The dataset creation repository can be found [here](https://github.com/iMEdD-Lab/Greek_Parliament_Proceedings).  
You can download directly the dataset in csv format using the [link](https://zenodo.org/record/4311577/files/Greek_Parliament_Proceedings_1989_2020.zip?download=1). 

## Spark Project
In the directory `spark-analysis` you can find the Spark Project we have created for the analysis of the aforementioned dataset, which we conducted using NLP techniques in Spark with Scala. 

### Reproduce project
This project was created using IntelliJIdea IDE with Scala 2.12.8 and Spark 3.0.0. In order to reproduce it you simple have to open the spark-analysis directory using IntelliJIdea and the required sbt dependencies will be build automatically. When project is ready you also need to add the csv file of the Greek Parliament Proceedings, 1989-2020 dataset in the `resources` directory under **spark-analysis/src/main** you will also have to rename it as **gpp.csv**.
