# personnal_project

## Idées en vrac

L'objectif de ce projet est de créer un modèle de classification d'actions en 3 classes : achetée, vendre, garder.
Mon idée est d'entrainer un modèle de regression logistique sur des données boursières historiques puis je me focalise sur certaines actions dont je récupère les features à l'instant présent pour voir ce que mon classifieur ferait.

## à faire 

- Réfléchir aux features sur lesquels travailler
- Trouver un dataSet possédant des features intéressantes
- Si je ne trouve pas ce dataset, pourquoi pas le construire à partir de données historiques ? 


- J'ai construit le dataset, maintenant il faut vérifier si les labels paraissent cohérents en fonction du prix des actions 

- il y a un problème de labelisation quand je backtest. réfléchir à comment est ce qu'on peut labeliser ? 

- il s agit de repondre à la question : si je connais le cours d une action à l avance, quand faut il acheter ou vendre ?

## Plans

- définition des labels
- définition des features
- construction du dataset
- est ce que la logique des labels établis dans le dataset est bonne ? i.e. est ce qu on gagne en suivant les labels ?
- entrainer la régression logistique
- évaluer la régression

## Fonctionnement des labels

Les labels sont déterminés à partir du retour futur (une valeur qui nous dit si on a gagné ou perdu)