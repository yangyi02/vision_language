### Introduction ###

This is a demo illustrating the caption generation given a pre-trained image captioning model. To run the demo, simply modify the PADDLE_PATH in demo.sh, and run demo.sh. 

```bash
bash demo.sh
```

After successfully running demo.sh, you should see a file called result.txt with contents below:
```bash
153529	 #OOV#	 a group of people standing around a luggage bag $$E$$
406491	 #OOV#	 a cat sitting on top of a wooden floor $$E$$
325211	 #OOV#	 a computer keyboard sitting on top of a desk $$E$$
13333	 #OOV#	 a red stop sign sitting on the side of a road $$E$$
268396	 #OOV#	 a stop sign in front of a tall building $$E$$
518836	 #OOV#	 a brown horse standing next to a brown horse $$E$$
251094	 #OOV#	 a couple of sheep standing next to each other $$E$$
492609	 #OOV#	 a pizza sitting on top of a wooden table $$E$$
453302	 #OOV#	 a kitchen with a sink and a refrigerator $$E$$
544198	 #OOV#	 a kitchen with a stove and a stove $$E$$
67956	 #OOV#	 a stop sign on the corner of a street $$E$$
```
Where the first column is image id, second column is question (#OOV# when it's image captioning task), third column is caption / answer. $$E$$ is the end sign of a sentence.

### Directory Explanation ###

* batch_0: a data batch containing image features
* conf.py: paddle python configure for generating text
* data_provider.py: paddle python data provider for generating text
* demo.sh: main running script
* dict.pkl: pickle file holding word dictionary
* dict.txt: text file holding word dictionary
* file.list: a file list store data batches
* model: directory holding Paddle model weights
