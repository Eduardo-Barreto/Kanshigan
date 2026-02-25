= Referencial teórico

== Visão computacional em análise esportiva

A visão computacional tem sido aplicada de forma crescente à análise esportiva. Sistemas de tracking de jogadores utilizando detectores baseados em YOLO combinados com trackers como DeepSORT @wojke2017deep tornaram-se padrão em futebol e basquete. Sistemas de pontuação automatizada foram desenvolvidos para taekwondo com abordagens human-in-the-loop, e sistemas de arbitragem eletrônica para tênis de mesa atingiram 97.8% de acurácia na detecção de posição da bola. Frameworks de action spotting como SoccerNet @giancola2018soccernet permitem detecção automática de eventos-chave em vídeos de partidas.

== Detecção de objetos e multi-object tracking

A detecção de objetos em tempo real é dominada pela família YOLO @jocher2023yolo, com versões recentes atingindo bom trade-off entre acurácia e velocidade. Abordagens baseadas em transformers como RT-DETR @zhao2024rtdetr oferecem detecção end-to-end sem supressão de não-máximos. Para multi-object tracking, ByteTrack @zhang2022bytetrack associa detecções de alta e baixa confiança para manter continuidade do rastreamento, enquanto BoT-SORT @aharon2022botsort incorpora features de aparência para maior robustez em oclusões.

== Visão computacional em competições de robôs

Na RoboCup, as ligas SSL (Small Size League) e VSSS (Very Small Size Soccer) utilizam sistemas de visão computacional com câmeras aéreas para detectar e rastrear robôs em tempo real. O SSL-Vision @zickler2010sslvision é um sistema compartilhado de visão que processa imagens de múltiplas câmeras overhead, identificando robôs por padrões coloridos e convertendo coordenadas de pixels para espaço real. Sobre esse sistema, Zhu e Veloso @zhu2017refereeing propuseram um árbitro automatizado baseado em eventos para partidas da SSL, validado contra decisões de árbitros humanos na RoboCup 2014.

No lado dos robôs de combate, a equipe do robô Orbitron (BattleBots) integrou YOLOv8 para detecção autônoma de oponentes. Um pequeno dataset de imagens de robôs de sumô existe no Roboflow Universe para fins de detecção, porém com robôs amadores que não refletem a categoria competitiva atual. Contudo, *nenhum trabalho anterior aborda a análise externa de partidas de Sumô de Robôs por vídeo* para extração de desempenho e classificação de eventos. Diferentemente da SSL/VSSS, onde câmeras são fixas e controladas, o Sumô de Robôs apresenta o desafio adicional de funcionar com vídeos de qualidade heterogênea e ângulos variáveis.

== Foundation models para segmentação

Modelos foundation de segmentação como SAM @kirillov2023sam e SAM 2 @ravi2024sam2 permitem segmentação zero-shot em imagens e vídeos. SAM 3 @carion2025sam3 introduz Promptable Concept Segmentation: dado um prompt textual ou visual, o modelo detecta, segmenta e rastreia todas as instâncias do conceito em imagens e vídeos, com ganho de 2x sobre sistemas anteriores nessa tarefa. Esses modelos são candidatos para anotação semi-automática do dataset, acelerando o processo de construção sem exigir labeling manual extensivo. Resultados preliminares no domínio de sumô de robôs indicam viabilidade dessa abordagem.
