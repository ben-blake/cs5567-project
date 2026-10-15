# Presentation script

Slide-by-slide, with a speaker and a time budget for each.

**The honest arithmetic:** 15 minutes including questions means about 13 minutes
of talking. Four people at a strict 4 minutes each is 16 minutes, which does not
fit. The budgets below total **13:30**, which leaves a little over a minute for
questions. Ben carries more because he opens and closes.

| Speaker | Slides | Time |
| --- | --- | --- |
| Ben (opening) | Title to 6 | 3:50 |
| Geethika | 7 to 9 | 2:30 |
| Tina | 10 to 12 | 2:40 |
| Sanjana | 13 to 16 | 2:45 |
| Ben (closing) | 17 to 20 | 1:45 |

This is a script, not a teleprompter. Learn the shape of each slide and the
numbers you must not get wrong; say it in your own words on the day.

---

## Ben, opening (3:50)

### Title slide | 10 seconds

> Good morning. We're Group 4, and our project is video object tracking on
> MOT16. I'm Ben, and I'll hand over to each of the others as we go.

### Slide 1, Team | 20 seconds

> Each of us owns one part of the pipeline end to end: the code, these slides,
> and our section of the final report. I took detection and tracking, Geethika
> the data pipeline, Tina re-identification, and Sanjana evaluation and the
> demo. Each of us will present our own piece.

### Slide 2, Problem | 35 seconds

> The task is this. Given a video, put a box around every pedestrian in every
> frame, and keep the same identity number on the same person throughout.
>
> Detection alone does not get you there. A detector answers "where are the
> people" separately in each frame. It has no idea that the person in frame 10
> is the person it saw in frame 9. Tracking is what adds that link, and it has
> to survive occlusion, people crossing in front of each other, and frames
> where the detector simply misses someone.
>
> Our target from the rubric: track at least one pedestrian across a sequence
> without losing them.

### Slide 3, System Architecture | 40 seconds

> The pipeline has three stages. Faster R-CNN finds people and gives us boxes.
> The Siamese network crops each person and turns them into an appearance
> vector. Then association decides which detection continues which track, and
> that is what produces stable IDs.
>
> Two design choices worth naming. We track offline, processing a batch of
> frames rather than streaming, which lets us recover from short gaps. And we
> use tracking by detection, keeping detection and identity separate, so either
> component can be swapped without touching the other.

### Slide 4, Detection: Faster R-CNN | 40 seconds

> We use Faster R-CNN with a ResNet50 FPN backbone. The Region Proposal Network
> generates proposals inside the network, so there is no external Selective
> Search, and proposals and detection share one backbone, so each image is
> convolved once. It is also accurate on small objects, which matters because
> distant pedestrians are small.
>
> For fine-tuning we freeze the backbone and train only the detection heads.
> That is 14.5 million trainable parameters out of 41.3 million, about 35%,
> which keeps the job inside Colab's memory budget.

> **If asked why not YOLO:** accuracy on small objects matters more than speed
> here. We run offline at about 4 FPS and are not trying to be real-time.

### Slide 5, What the detector actually sees | 40 seconds

> This is frame 1 of MOT16-02. Ground truth on top, our detections below, with
> the COCO-pretrained model and no fine-tuning yet.
>
> It finds the people in the foreground reliably. What it misses are the small,
> distant and partly hidden ones, and there are a lot of those in a crowded
> street. Our recall is around 0.40, so we find roughly four in ten of the
> annotated people.
>
> Hold on to that number. It turns out to explain almost everything else we
> found.

### Slide 6, Association Problem | 45 seconds

> Every frame gives us some detections, and we are already holding some tracks.
> The question is which detection continues which track.
>
> Overlap alone fails: when two people cross, each box overlaps where the other
> one used to be, so position cannot separate them. Appearance alone also
> fails, because two people in similar clothing produce similar vectors, and
> appearance drifts as someone turns.
>
> So we blend them. Lambda trades off box overlap against appearance distance,
> and we solve the assignment with the Hungarian algorithm, which gives a
> globally optimal one-to-one matching rather than greedy nearest-neighbour.
> Pairs that fail either gate are rejected before matching.
>
> Geethika is going to talk about the data this all runs on.

---

## Geethika, data pipeline (2:30)

### Slide 7, The MOT16 Dataset | 45 seconds

> MOT16 is fourteen sequences of crowded streets: seven training with ground
> truth, seven test without. Moving and static cameras, heavy occlusion, people
> constantly leaving and re-entering the frame.
>
> The annotations are plain CSV, one box per line. Frame number, identity,
> then the box as left, top, width and height. After that a confidence flag, a
> class, and a visibility fraction.
>
> The sequence we use for measurement is MOT16-02: 600 frames, 54 identities,
> about 17,800 annotated boxes.

### Slide 8, Two traps in the annotations | 1:00

> Two things in this format will silently wreck your results, and we hit both.
>
> First, coordinates are 1-indexed. The files count pixels from 1, NumPy counts
> from 0. If you miss it, every box is off by one pixel. Nothing looks wrong:
> the video still shows boxes on people. But IoU drops slightly everywhere, and
> your metrics get worse for no visible reason.
>
> Second, not every annotated person is a target. Classes 7, 8 and 12 are
> static people, distractors and reflections. They are people, but the benchmark
> does not score them. If you train on them you are teaching the detector to
> fire on mannequins and reflections in windows.
>
> Both are handled in our parsing code, and we wrote the unit tests first,
> precisely because neither failure announces itself.

### Slide 9, Data Augmentation | 45 seconds

> MOT16 gives us a small number of scenes, so the real risk is that the model
> memorises those scenes instead of learning what a pedestrian looks like.
>
> For detection we use horizontal flips, with the boxes mirrored to match, and
> colour jitter on brightness, contrast and saturation. For the re-ID crops we
> add random cropping and random erasing, which blanks a rectangle to simulate
> occlusion.
>
> What we deliberately leave out is vertical flips and large rotations.
> Pedestrians are upright in every frame of every sequence, so an upside-down
> person is a pose the model will never see at test time. Training on it just
> spends capacity on something useless.
>
> Tina will take the re-identification side.

---

## Tina, re-identification (2:40)

### Slide 10, Re-identification: a similarity model, not a classifier | 55 seconds

> Re-identification is the component that answers: are these two crops, from
> different frames, the same person?
>
> The obvious approach is a classifier over the people in the video. That does
> not work here, because every new person who walks into the scene would need
> the model retrained, and MOT16 sequences are full of people walking in.
>
> So instead we learn an embedding. Each crop becomes a 128-dimensional unit
> vector, trained so that crops of the same person land close together and
> crops of different people land apart. The network learns similarity rather
> than a fixed list of identities, so a new person needs no retraining at all.
>
> The architecture is a Siamese network: one CNN applied twice, with shared
> weights. Four convolution blocks, adaptive pooling, then the embedding. About
> 1.5 million parameters, so it trains in minutes.
>
> We train it with contrastive loss, which pulls matching pairs together and
> pushes non-matching pairs apart until they clear a margin.

### Slide 11, Training the similarity model | 45 seconds

> Eight epochs, on crops cut from MOT16's own ground truth, so the training
> data matches the footage we actually track.
>
> On the left, the loss falls steadily and validation accuracy on held-out
> pairs rises from 0.535, which is chance, to 0.932.
>
> On the right is the same set of crops embedded before and after training.
> Before, the similarity matrix is nearly uniform: the network cannot tell
> anyone apart. After, you can see block structure along the diagonal, which is
> the network having learned who is who.

### Slide 12, A problem we had to resolve | 1:00

> This one is worth describing because it is the kind of thing you only find by
> trying to implement from the spec.
>
> The course slides give four different crop sizes. Training is specified at
> 16 by 16, inference at 24 by 24, retrieved crops at 30 by 16, and the
> reference Siamese network code has a linear layer that actually requires
> about 102 by 102. No single size satisfies all four.
>
> We chose 128 by 64, and we can justify it: that is the resolution the
> Market-1501 paper normalises its own images to, and Market-1501 is the
> standard person re-identification benchmark. Sixteen by sixteen simply cannot
> resolve clothing detail, which is the entire signal re-identification depends
> on.
>
> We also replaced the hardcoded flatten with adaptive pooling, so the network
> accepts any input size and the choice is no longer baked into the
> architecture.
>
> Sanjana will cover how we measure all this.

---

## Sanjana, evaluation (2:45)

### Slide 13, How we measure success | 40 seconds

> We use the standard CLEAR-MOT metrics. MOTA is one minus the sum of false
> negatives, false positives and identity switches, divided by the number of
> ground-truth boxes. So it penalises people we miss, boxes with nobody in
> them, and the ID changing on one person. MOTP is separate: it is the mean IoU
> of the boxes we did match, so it measures how tightly they fit.
>
> Of the three error types, identity switches matter most to us. A missed frame
> can be recovered on the next one. An ID switch means we have confused two
> people, and that is exactly the failure re-identification exists to prevent.

### Slide 14, Results: first end-to-end run | 50 seconds

> This is MOT16-02, 600 frames, on a Colab T4, with the COCO-pretrained
> detector and no fine-tuning yet.
>
> Against the rubric: one real pedestrian is held by a single ID for 218 of the
> 230 frames they appear in, which is 94.8%. That is measured against the
> ground-truth person, not against whichever of our tracks happened to live
> longest, and that distinction matters.
>
> But the median across all 44 people visible for a hundred frames or more is
> 22.4%. One person is tracked well. Most are not.
>
> Overall MOTA is 0.232 and MOTP is 0.792. The MOTP tells us the boxes we
> produce are tight. The MOTA is dragged down almost entirely by false
> negatives: nearly 11,000 of them against 7,000 true positives.

### Slide 15, Coverage across every person | 25 seconds

> This is that median drawn out. Each bar is one ground-truth person, showing
> the best coverage any single predicted ID achieved for them.
>
> The one on the left is the 94.8% result. Everything after it falls away
> quickly. The honest reading is that we can track one person well and most
> people poorly, and closing that gap is a detection problem rather than a
> tracking problem.

### Slide 16, Does re-identification help? | 50 seconds

> Here is the result we did not expect. We ran the pipeline with appearance
> matching on and with it off, identical in every other respect.
>
> MOTA is identical. MOTP is identical. ID switches go from 151 to 148, which
> is slightly *worse* with re-identification than without. Our appearance
> matching is doing essentially nothing.
>
> The threshold sweep on the right explains why. As we lower the detection
> threshold, recall barely moves, from 0.40 to 0.51, while false positives
> quadruple. So the people we miss are not low-confidence detections being
> filtered out. They are people the COCO model does not find at all.
>
> And false negatives outnumber ID switches by about seventy to one. Detection
> is the bottleneck, so association has almost nothing left to disambiguate.
> That is our empirical case for fine-tuning the detector, which is the next
> thing we do.
>
> Ben has one more result to close on.

---

## Ben, closing (1:45)

### Slide 17, A bug we found, fixed, and then turned off | 1:00

> One more result, because it is the most interesting thing we learned.
>
> We produce 176 track IDs for 54 real people, which looks like a bug, and
> there genuinely was one. With an identical appearance vector, a five-frame
> gap was enough to lose the identity: the track's box stayed where the person
> used to be, and the overlap gate rejected the match before appearance was
> ever consulted.
>
> So we fixed it, and the fix made the tracking worse. Look at the bottom row.
> 54 IDs for 54 people looks perfect, but identity switches almost triple. The
> tracker is not giving each person their own ID. It is passing the same IDs
> around between different people.
>
> The reason is the recall number again. Only 40% of people are detected, so
> most apparent reappearances are strangers we never saw arrive, and our re-ID
> network is 93% accurate, not perfect. Splitting one person into several IDs
> costs you one interruption. Merging several people into one ID corrupts both
> trajectories. At this recall the cautious tracker wins, so both features ship
> disabled.

### Slide 18, Demo | 25 seconds

> Finally, here is what it looks like. An eight-second clip, with the target in
> colour and every other track in grey.

> **[Alt-tab to `demo_target_clip.mp4`. Let it play once without talking over
> it, then return to the slides.]**

### Slide 19, Timeline to the final | 20 seconds

> Where we are: the pipeline runs end to end, and everything you have seen is
> measured rather than estimated, with 78 tests in CI.
>
> What is next: fine-tuning the detector, because recall is what caps
> everything. Then we re-run the ablation, because our conclusion that
> re-identification does not help is a conclusion about *this* recall level,
> not a permanent one. Then the final video on a test sequence, and the report.

### Slide 20, Questions | remainder

> That is us. Happy to take questions.

---

## Handover lines

Keep them short. One sentence, naming the next person and their topic.

- Ben to Geethika: *"Geethika is going to talk about the data this all runs on."*
- Geethika to Tina: *"Tina will take the re-identification side."*
- Tina to Sanjana: *"Sanjana will cover how we measure all this."*
- Sanjana to Ben: *"Ben has one more result to close on."*

## Numbers you must not get wrong

| Claim | Value |
| --- | --- |
| Target coverage | 94.8%, 218 of 230 frames |
| Median coverage | 22.4% across 44 people |
| MOTA / MOTP | 0.232 / 0.792 |
| Recall | about 0.40 |
| Re-ID accuracy | 0.535 to 0.932 |
| Trainable parameters | 14.5 M of 41.3 M, 35% |
| Unique IDs vs real people | 176 vs 54 |
| Tests | 78 |

**Never say 97.2%.** An earlier version of our deck used it; it came from the
longest-lived track and moved to 73.0% on re-run with nothing else changing.
94.8% is the number that holds still.

## If you are running long

Cut in this order. Each is self-contained, so dropping it leaves no dangling
reference:

1. Slide 9, Data Augmentation. Geethika mentions it in one line on slide 8.
2. Slide 15, Coverage across every person. The median is already stated on
   slide 14.
3. Slide 3, System Architecture. The components are introduced as they come up.

Do **not** cut slides 12, 16 or 18. Those three are the "understanding on the
methods" the assignment is asking for, and they are what separates this from a
description of a tutorial.
