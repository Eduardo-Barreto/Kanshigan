= 1 CONTEXT

Autonomous Robot Sumo competitions (3kg class) are high-speed mobile robotics contests where entire rounds last fractions of a second. With approximately 80,000 active competitors across more than 30 countries, the sport has a significant global community participating in tournaments such as the All Japan Robot Sumo Tournament and regional competitions like RoboCore (Brazil) and RoboGames (USA).

Despite this scale, all performance analysis remains entirely manual and subjective. No automated tools, structured databases, or standardized metrics exist for the domain. The absence of quantitative data directly affects three groups:

- *Competitive robotics teams* lack real combat data to support design decisions, building new robots based on intuition and memory.
- *Competition organizers* have no structured historical documentation or objective analysis tools.
- *Spectators* watch live broadcasts without any performance statistics to enrich the viewing experience. In contrast with traditional sports (soccer, baseball, basketball), where real-time statistics are a fundamental part of broadcasting, Robot Sumo offers zero data to the viewer.

In the computer vision literature, sports analysis has advanced significantly in domains such as soccer, combat sports, and table tennis. However, no work addresses automated external analysis of Robot Sumo matches. The domain presents unique technical challenges: extreme speed (rounds lasting fractions of a second), small and visually similar objects, mutual occlusion during contact, and heterogeneous video quality ranging from professional broadcasts to amateur recordings.
