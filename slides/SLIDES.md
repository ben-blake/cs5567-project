# Speaking notes

One entry per slide: who talks, roughly how long, and what to say.

Fifteen minutes has to cover questions too, so we've got about thirteen and a
half minutes of actual talking. Four minutes each would be sixteen, which
doesn't fit. Ben takes a bit more because he opens and closes.

| Speaker       | Slides     | Time |
| ------------- | ---------- | ---- |
| Ben (opening) | Title to 6 | 3:50 |
| Geethika      | 7 to 9     | 2:30 |
| Tina          | 10 to 12   | 2:40 |
| Sanjana       | 13 to 16   | 2:45 |
| Ben (closing) | 17 to 20   | 1:45 |

Don't memorise this. Learn what each slide is for and the numbers you can't get
wrong, then say it your own way.

---

## Ben, opening (3:50)

### Title | 10 sec

> Morning. We're Group 4, and we've been working on video object tracking on
> MOT16. I'll start, and we'll each take our own part.

### Slide 1, Team | 20 sec

> We split this four ways, and each of us owns a piece end to end: the code,
> the slides, and our section of the report. I did detection and tracking,
> Geethika the data pipeline, Tina re-identification, Sanjana evaluation and
> the demo.

### Slide 2, Problem | 35 sec

> The task is: given a video, draw a box around every pedestrian in every
> frame, and keep the same ID on the same person the whole way through.
>
> A detector on its own won't do that. It answers "where are the people" one
> frame at a time, and it has no idea the person in frame 10 is the one it saw
> in frame 9. Linking those together is the tracking part, and it has to cope
> with people walking behind each other, crossing, and frames where the
> detector just misses someone.
>
> What we're aiming for, from the rubric, is one pedestrian tracked across a
> sequence without losing them.

### Slide 3, System Architecture | 40 sec

> Three stages. Faster R-CNN finds people and gives us boxes. The Siamese
> network crops each person out and turns them into an appearance vector. Then
> association works out which detection continues which track, and that's what
> gives you stable IDs.
>
> Two things worth pointing out. We're running offline, on a batch of frames
> rather than a live stream, which means we can recover from short gaps. And
> detection and identity are kept separate, so we can swap either one without
> touching the other.

### Slide 4, Detection: Faster R-CNN | 40 sec

> We went with Faster R-CNN on a ResNet50 FPN backbone. The Region Proposal
> Network generates proposals inside the network, so we're not running
> Selective Search separately, and the proposals and the detector share one
> backbone, so each image only gets convolved once. It also does well on small
> objects, which matters here because people further down the street are small.
>
> For fine-tuning we freeze the backbone and only train the heads. That's 14.5
> million trainable parameters out of 41.3 million, about 35%, which fits
> comfortably in Colab.

> **If someone asks why not YOLO:** for offline tracking we care more about
> accuracy on small objects than speed. We run at about 4 FPS and we're not
> trying to be real-time.

### Slide 5, What the detector actually sees | 40 sec

> This is frame 1 of MOT16-02. Ground truth on top, what we detect underneath,
> using the COCO-pretrained model with no fine-tuning yet.
>
> It gets the people in the foreground. What it misses are the small, distant
> and partly hidden ones, and a crowded street is full of those. Our recall is
> around 0.40, so we're finding about four in ten of the annotated people.
>
> That number comes back later.

### Slide 6, Association Problem | 45 sec

> Each frame gives us some detections, and we're already holding some tracks.
> We need to decide which detection continues which track.
>
> Overlap on its own breaks down when two people cross, because each box
> overlaps where the other one just was. Appearance on its own isn't enough
> either, since two people in similar clothes give you similar vectors, and
> appearance shifts as someone turns around.
>
> So we combine them. Lambda controls how much weight goes on box overlap
> versus appearance distance, and we solve the matching with the Hungarian
> algorithm, which gets a globally optimal one-to-one assignment instead of
> just taking the nearest match each time.
>
> Geethika's going to talk about the data underneath all this.

---

## Geethika, data pipeline (2:30)

### Slide 7, The MOT16 Dataset | 45 sec

> MOT16 is fourteen sequences shot on busy streets. Seven have ground truth and
> are for training, seven are test. Mix of moving and static cameras, a lot of
> occlusion, people walking in and out of frame constantly.
>
> The annotations are just CSV, one box per line: frame number, person ID, then
> the box as left, top, width, height, and after that a confidence flag, a
> class, and how visible the person is.
>
> We measure on MOT16-02, which is 600 frames with 54 people and about 17,800
> boxes.

### Slide 8, Two traps in the annotations | 1:00

> There are two things in this format that'll quietly ruin your results, and we
> hit both of them.
>
> The first is that coordinates start at 1, not 0. The files count pixels from
> 1 and NumPy counts from 0, so if you miss it every box is off by a pixel.
> Nothing looks broken. The video still shows boxes sitting on people. But your
> IoU is slightly low everywhere and your metrics are worse for no reason you
> can see.
>
> The second is that not everyone annotated is meant to be tracked. Classes 7,
> 8 and 12 are static people, distractors and reflections. They're people, but
> the benchmark doesn't score them, and if you train on them you're teaching
> the detector to fire on mannequins and shop windows.
>
> Both are handled in our parsing code. We wrote the tests for them first,
> because neither one gives you an obvious symptom.

### Slide 9, Data Augmentation | 45 sec

> We've only got a handful of scenes, so the risk is the model memorises those
> scenes rather than learning what a pedestrian looks like.
>
> For detection we use horizontal flips, with the boxes mirrored to match, plus
> colour jitter on brightness, contrast and saturation. For the re-ID crops we
> add random cropping and random erasing, which blanks out a rectangle to
> mimic occlusion.
>
> We deliberately left out vertical flips and big rotations. Pedestrians are
> upright in every frame of every sequence, so an upside-down person is a pose
> the model is never going to see.
>
> Tina's got the re-identification side.

---

## Tina, re-identification (2:40)

### Slide 10, Re-identification: a similarity model, not a classifier | 55 sec

> Re-identification answers one question: are these two crops, from different
> frames, the same person?
>
> The obvious thing is to train a classifier over the people in the video, and
> that doesn't work here. Every time someone new walks into the shot you'd have
> to retrain, and these sequences are nothing but people walking into the shot.
>
> So we learn an embedding instead. Each crop becomes a 128-dimensional unit
> vector, and we train it so the same person's crops end up close together and
> different people end up far apart. It learns similarity rather than a list of
> identities, which means a new person needs no retraining at all.
>
> It's a Siamese network, so one CNN applied twice with shared weights. Four
> convolution blocks, adaptive pooling, then the embedding. About 1.5 million
> parameters, so it trains in a few minutes.
>
> The loss is contrastive loss: pull matching pairs together, push the rest
> apart until they clear a margin.

### Slide 11, Training the similarity model | 45 sec

> Eight epochs on crops cut out of MOT16's own ground truth, so what it trains
> on looks like what it'll see.
>
> On the left the loss comes down steadily, and accuracy on held-out pairs goes
> from 0.535, which is chance, up to 0.932.
>
> On the right is the same set of crops before and after training. Before, the
> similarity matrix is more or less uniform; it can't tell anyone apart. After,
> you can see blocks forming along the diagonal, and that's it having learned
> who's who.

### Slide 12, A problem we had to resolve | 1:00

> This one's worth mentioning because it's the sort of thing you only run into
> once you actually try to build from the spec.
>
> The course slides give four different crop sizes. Training is 16 by 16,
> inference is 24 by 24, the retrieved crops are 30 by 16, and the reference
> Siamese code has a linear layer that needs roughly 102 by 102. There's no
> size that satisfies all four.
>
> We went with 128 by 64, and we can point to a reason: that's what the
> Market-1501 paper normalises its images to, and Market-1501 is the standard
> benchmark for this. At 16 by 16 you can't make out clothing at all, and
> clothing is basically the whole signal re-identification runs on.
>
> We also swapped the hardcoded flatten for adaptive pooling, so the network
> takes any input size and we're not locked into the choice.
>
> Sanjana's going to cover how we measure all this.

---

## Sanjana, evaluation (2:45)

### Slide 13, How we measure success | 40 sec

> We're using the standard CLEAR-MOT metrics. MOTA is one minus false
> negatives plus false positives plus ID switches, over the number of
> ground-truth boxes, so it penalises people we miss, boxes with nobody in
> them, and the ID changing on someone. MOTP is separate and simpler: the mean
> IoU of the boxes we matched, so it's about how tight they are.
>
> Of the three errors, ID switches are the ones we care about most. A missed
> frame you get back on the next one. An ID switch means we've mixed up two
> people, and that's exactly what re-identification is supposed to stop.

### Slide 14, Results: first end-to-end run | 50 sec

> MOT16-02, 600 frames, on a Colab T4, with the pretrained detector and no
> fine-tuning yet.
>
> On the rubric: one real person is held by a single ID for 218 of the 230
> frames they're in, so 94.8%. That's measured against the actual ground-truth
> person, not against whichever of our tracks happened to survive longest,
> which turns out to matter.
>
> The median across all 44 people who are on screen for a hundred frames or
> more is 22.4%. So one person we track well, most we don't.
>
> MOTA overall is 0.232, MOTP is 0.792. The MOTP says the boxes we do produce
> are tight. The MOTA is low almost entirely because of false negatives: close
> to 11,000 of them against 7,000 true positives.

### Slide 15, Coverage across every person | 25 sec

> This is that median spelled out. Each bar is one ground-truth person, and the
> height is the best any single one of our IDs managed for them.
>
> The bar on the left is the 94.8%. After that it drops off fast. Which is to
> say we can follow one person well and most people badly, and the gap there is
> a detection problem, not a tracking one.

### Slide 16, Does re-identification help? | 50 sec

> This is the result we weren't expecting. We ran the whole thing with
> appearance matching on, then with it off, everything else identical.
>
> MOTA is the same. MOTP is the same. ID switches go from 151 to 148, so
> slightly worse with re-identification than without. It's doing essentially
> nothing.
>
> The sweep on the right shows why. As we drop the detection threshold, recall
> barely moves, 0.40 to 0.51, while false positives go up four times over. The
> people we're missing aren't low-confidence detections getting filtered out,
> they're people the COCO model doesn't find at all.
>
> And false negatives outnumber ID switches roughly seventy to one. Detection
> is the bottleneck, so there's almost nothing left for association to
> disambiguate. That's our argument for fine-tuning the detector, and it's what
> we're doing next.
>
> Ben's got the demo to finish on.

---

## Ben, closing (1:45)

### Slide 17, Demo | 25 sec

> And here's what it looks like. Eight seconds, target in colour and
> everything else in grey.

> **[Switch to `demo_target_clip.mp4`. Let it run without talking over it, then
> > come back to the slides.]**

### Slide 18, Timeline to the final | 20 sec

> Where we are: it runs end to end, everything we've shown is measured rather
> than estimated, and there are 78 tests behind it.
>
> Next is fine-tuning the detector, since recall is what's capping everything
> else. Then we re-run the ablation, because "re-identification doesn't help"
> is a statement about this recall level and not a permanent one. Then the
> final video on a test sequence, and the report.

---

