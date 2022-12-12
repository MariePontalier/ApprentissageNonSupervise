# ApprentissageNonSupervise

Le code python se trouve dans : 
ApprentissageNonSupervise/clustering-benchmark-master/src/main/resources/datasets/python/


Pour la première partie des TPs, nous avons fait un fichier par méthode.
Dans chaque fichier nous avons mis en haut quelques datasets avec leurs meilleurs paramètres correspondant.
Pour exécuter la méthode sur un dataset il suffit de décommenter cette partie et exécuter le fichier python:

```
# databrut = arff.loadarff(open(path+"/xclara.arff", 'r'))
# k = 3
```

Egalement nous avons utilisé des boucles for pour tester ces méthodes avec différents paramètres et renvoyer
le meilleur résultat selon les métriques d'évaluation. Ces boucles for sont commentés dans les fichiers. Normalement il suffit
de les décommenter pour les utiliser.

Le fichier Main.py correspond à la deuxième partie des TPs où nous avons lancer les méthodes de clustering sur chacun des nouveaux datasets.
Les fonctions methodBasic correspondent à une itération de la méthode où nous donnons les paramètres alors que les fonctions
methodAuto essayent de trouver les meilleurs paramètres avec les métriques d'évaluation.

Il faut décommenter les lignes correpsondant à la méthode que vous souhaiter lancer sur le datatset correspondant.
