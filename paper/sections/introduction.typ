= Introduction

// TODO: Expandir introdução com base no projeto de pesquisa v1

Robot Sumo competitions are high-speed autonomous robotics events where two robots attempt to push each other out of a circular arena (dohyo) within fractions of a second. With approximately 80,000 active competitors across over 30 countries, the sport has a large and engaged global community participating in tournaments such as the All Japan Robot Sumo Tournament @fujisoft-about and regional competitions like RoboCore (Brazil) and RoboGames (USA) @robogames-rules.

Despite this scale, all performance analysis remains entirely manual and subjective. No automated tools, structured databases, or standardized metrics exist for the domain. This absence of quantitative data affects three key stakeholder groups:

+ *Competitive teams* lack real combat data to inform design decisions, relying on intuition and memory rather than evidence.
+ *Competition organizers* have no structured historical documentation or objective analysis tools.
+ *Spectators* watch live broadcasts without any performance statistics, limiting engagement and comprehension of what differentiates a winning robot --- a stark contrast to traditional sports where real-time analytics are integral to the viewing experience.

In the computer vision literature, sports analytics has seen significant advances in domains such as soccer @giancola2018soccernet, combat sports, and table tennis. However, no prior work addresses automated external analysis of Robot Sumo matches. The domain presents unique technical challenges: extreme speed (sub-second rounds), small visually similar objects, mutual occlusion during contact, and heterogeneous video quality ranging from professional broadcasts to amateur recordings.

This paper makes the following contributions:
+ *Kanshigan*, the first open-source computer vision pipeline for automated analysis of Robot Sumo matches.
+ The first publicly available annotated dataset of 3kg autonomous Robot Sumo matches.
+ A comparative benchmark of detection and tracking approaches applied to this novel domain.
+ A set of standardized quantitative performance metrics for Robot Sumo (position, velocity, acceleration, reaction time).
