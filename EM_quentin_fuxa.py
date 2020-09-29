#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
L'objectif est de calculer les performances d'un algorithme EM, en fonction :
- Du nombre de possibilités de la variable cachée (= nb de cluster)
- De la taille des données (= nombre de réalisations de la variable aléatoire)
- De la quantité de données pour une réalisation (= taille du vecteur de features pour une réalisation)
- Du nombre de possibilité pour un feature d'une personne (= taille du vecteur de possibilités pour un feature)

On prend l'expérience classique d'un questionnaires, avec 
- Plusieurs questions (= taille du vecteur de features pour une réalisation)
- Plusieurs réponses possibles par questions (= taille du vecteur de possibilités pour un feature)
- Plusieurs personnes qui répondent (= nombre de réalisations de la variable aléatoire)
- Plusieurs groupes de personnes à identifier (= nb de cluster, il s'agit de la variable cachée)
On considère déjà savoir le groupes à identifier, mais on pourrait également appliquer une méthode de type Elbow comme
pour les algorithmes de clustering plus conventionnels, comme les kmeans.

'''


import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import argparse


def basicModel():
    '''  Crée un modèle pour les tests
        
        Renvoie 
        -------
        (mixtureCoefs, probs) où
        - mixtureCoefs est un tableau à 1 dimension contenant les coefficients de mélange
        - probs est un tableau 3D de taille (nClusters, nQuestions, nAnswers)
                contenant les distributions de Dirichlet des réponses
                pour chaque couple (cluster, question)
        '''

        # Mélange de deux clusters

    mixtureCoefs = np.array([0.4, 0.6])

        # Distribution des 3 réponses possibles pour chacune des 4 questions pour les deux clusters

    probs = np.array([[[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.7, 0.1,0.2], [0.3, 0.3, 0.4]], 
    [[0.7, 0.1, 0.2], [0.2,0.6, 0.2], [0.2, 0.4, 0.4], [0.2, 0.6, 0.2]]])  # Cluster 1
                                                                     # Cluster 2

        # Le modèle est

    return (mixtureCoefs, probs)


def generateModel(nQuestions, nAnswers, nClusters):
    ''' Génère un modèle de mélange de distributions de Dirichlet.
        
        Paramètres
        ----------
        nQuestions :    int, nombre de questions
        nAnswers :      int, nombre de réponses
        nClusters :     int, nombre de clusters
        
        Renvoie 
        -------
        (mixtureCoefs, probs) où
        - mixtureCoefs est un tableau à 1 dimension contenant les coefficients de mélange
        - probs est un tableau 3D de taille (nClusters, nQuestions, nAnswers)
                contenant les distributions de Dirichlet des réponses
                pour chaque couple (cluster, question)
        '''

    mixtureCoefs = np.random.dirichlet(np.ones(nClusters))
    probs = np.random.dirichlet(np.ones(nAnswers), (nClusters,
                                nQuestions))
    return (mixtureCoefs, probs)


def generateData(model, nPeople):
    ''' Génère des réponses au questionnaire à partir d'un modèle de mélange
                
        Paramètres
        ----------
        model :         modèle utilisé pour générer les données
        nPeople :       int, nombre de personnes interrogées

        Renvoie
        -------
        (data, cluster) où
        - data est un tableau 2D de taille (nPeople, nQuestions) contenant les réponses allant de 1 à nAnswers pour chacune des personnes
        - cluster est un tableau 2D de taille (nPeople, nbCluster). cluster[i,j] contient 1 si la personne i appartient au cluster j, 0 sinon.
        '''

    (mixtureCoefs, probs) = model
    (nClusters, nQuestions, nAnswers) = probs.shape
    clusters = np.random.choice(nClusters, size=nPeople, p=mixtureCoefs)
    data = np.zeros((nPeople, nQuestions))
    answers = range(1, nAnswers + 1)
    for i in range(nPeople):
        for j in range(nQuestions):
            data[i, j] = np.random.choice(answers, p=probs[clusters[i],
                    j, :])

    C = np.zeros((nPeople, len(model[1])))
    for (ind, cluster) in enumerate(clusters):
        C[ind, cluster] = 1
    return (np.asarray(data, 'uint'), C)


def drawModel(model, title=None):
    ''' Trace les distributions des réponses pour tous les clusters
        
        Paramètres
        ----------
        model :         modèle dont on veut tracer les distributions
        '''

    plt.ion()
    (_, probs) = model
    (nClusters, nQuestions, nAnswers) = probs.shape
    questions = np.arange(nQuestions)
    fig = plt.figure()
    fig.suptitle(title)
    for cluster in range(nClusters):
        plt.subplot(nClusters, 1, cluster + 1)
        colors = itertools.cycle([
            'b',
            'g',
            'r',
            'c',
            'm',
            'y',
            'k',
            ])
        total = np.zeros(nQuestions)
        for answer in range(nAnswers):
            c = next(colors)
            d = probs[cluster][:, answer]
            plt.bar(questions, d, width=1, bottom=total, color=c)
            total += d
        plt.yticks(np.arange(0, 1.0001, 0.5))
        plt.ylim((0, 1))
        plt.title('Groupe ' + str(cluster + 1))

    plt.xlabel('Questions')
    fig.tight_layout()
    plt.show()


def drawCluster(H, title='Distribution des variables cachées'):
    ''' Trace les distributions des variables cachées (ie l\appartenance à chacun des clusters ), pour chacune des personnes intérrogées
        
        Paramètres
        ----------      
        H: matrice dont le nombre de colonnes est égal au nombre de clusters et le coefficient H[i,j] donne la probabilité que 
                la personne i appartient au cluster j
        '''

    plt.ion()
    if len(H.shape) >= 2:
        (nPeople, nClusters) = H.shape
    else:
        (nPeople, ) = H.shape
        nClusters = np.max(H)
        Hnew = np.zeros((nPeople, nClusters))
        for (i, j) in zip(range(nPeople), H - 1):
            Hnew[i, j] = 1.
        H = Hnew
    fig = plt.figure()
    people = np.arange(nPeople)
    colors = itertools.cycle([
        'b',
        'r',
        'g',
        'c',
        'm',
        'y',
        'k',
        ])
    total = np.zeros(nPeople)
    for cluster in range(nClusters):
        c = next(colors)
        d = H[:, cluster]
        plt.bar(people, d, width=1, bottom=total, color=c)
        total += d
    plt.yticks(np.arange(0, 1.0001, 0.25))
    plt.ylim((0, 1))
    plt.title(title)
    plt.xlabel('Personnes', fontsize=14)
    fig.tight_layout()
    plt.show()


def compareClusters(C1, C2):
    ''' Détecter la correspondance des clusters dans C1 et C2, et donner le score pour la meilleur correspondance.
        
        Renvoie 
        -----
        la correspondance des clusters de C2 avec ceux de C1
                bestPerm[num du cluster de C2] = num du cluster de C1 correspondant

                bestOverlap correspond à l'accuracy obtenue pour la permutation optimale
        '''

    if len(C1.shape) >= 2 and C1.shape[1] > 1:
        C1 = np.argmax(C1, 1) + 1
    if len(C2.shape) >= 2 and C2.shape[1] > 1:
        C2 = np.argmax(C2, 1) + 1

    nClusters = max(np.max(C1), np.max(C2))
    perms = itertools.permutations(range(1, nClusters + 1))
    bestOverlap = 0.
    bestPerm = None
    for perm in perms:
        p = np.array(perm)
        permC2 = p[C2 - 1]
        overlap = np.sum(C1 == permC2) * 1. / np.size(C1)
        if overlap > bestOverlap:
            bestOverlap = overlap
            bestPerm = p #-1
    return (bestOverlap, bestPerm)


def permuteClusters(M, H, perm):
    ''' Permute les clusters pour que les couleurs des clusters
        dans les fonctions d'affichage correspondent
        Renvoie (permuted model, permuted H)
        '''

    invperm = np.argsort(perm - 1)
    (mixtureCoefs, probs) = M
    mixtureCoefs = mixtureCoefs[invperm]
    probs = probs[invperm, :, :]
    H = H[:, invperm]
    return ((mixtureCoefs, probs), H)


def em_non_opti(data, nClusters, nIterations):  # pas optimisé mais explicite
    ''' Applique l'algorithme EM pendant le nombre d'itérations nIterations
        pour trouver les paramètres de chacun des clusters nClusters du modèle des données de data
        
        Renvoie 
        ---------
        le couple (model,H) où
        - model est le modèle appris  à partir des données
        - H est une matrice (nombre de personnes, nombre de clusters) dont les lignes
        correspondent aux distributions de cluster de chaque personne
        '''

    (nPeople, nQuestions) = data.shape
    nAnswers = int(np.max(data))

        # Clusters equiprobables

    mixtureCoefs = np.ones(nClusters) / nClusters
    probs = np.random.dirichlet(np.ones(nAnswers), (nClusters,
                                nQuestions))
    H = np.zeros((nPeople, nClusters))

    for t in range(nIterations):
        for i in range(nPeople):
            answers = data[i, :]
            for c in range(nClusters):
                H[i, c] = np.log(mixtureCoefs[c])
                for j in range(nQuestions):
                    H[i, c] += np.log(probs[c, j, int(answers[j] - 1)])
            H[i, :] = H[i, :] - np.max(H[i, :])
            H[i, :] = np.exp(H[i, :])

            H[i, :] /= np.sum(H[i, :])

        weights = np.sum(H, 0)
        mixtureCoefs = weights / np.sum(weights)

        for cluster in range(nClusters):
            for j in range(nQuestions):
                for k in range(1, nAnswers + 1):
                    s = 0.
                    total = 0.
                    for i in range(nPeople):
                        if data[i, j] == k:
                            s += H[i, cluster]
                        total += H[i, cluster]
                        probs[cluster, j, k - 1] = s / total

                probs[cluster, j, :] /= sum(probs[cluster, j, :])

    return ((mixtureCoefs, probs), H)


def loglikelihood(model, data, H):
    '''  Calcule la vraisemblance
        '''

    (_, probs) = model
    (nPeople, nQuestions) = data.shape

    logL = 0.
    for i in range(nPeople):
        answers = data[i, :]
        for j in range(nQuestions):
            logL += np.log(sum(probs[:, j, int(answers[j]) - 1] * H[i, :
                           ].T))
    return logL


def em_opti(data, nClusters, epsilon):
    ''' Applique l'algorithme EM pendant le nombre d'itérations nIterations
        pour trouver les paramètres de chacun des clusters nClusters du modèle des données de data
        
        Renvoie 
        ---------
        le couple (model,H) où
        - model est le modèle appris  à partir des données
        - H est une matrice (nombre de personnes, nombre de clusters) dont les lignes
        correspondent aux distribution de cluster de chaque personne
        '''

    (nPeople, nQuestions) = data.shape
    nAnswers = int(np.max(data))

    mixtureCoefs = np.ones(nClusters) / nClusters
    probs = np.random.dirichlet(np.ones(nAnswers), (nClusters,
                                nQuestions))

    H = np.zeros((nPeople, nClusters))
    oldLogL = -sys.float_info.max

    while True:

        H = np.ones((nPeople, 1)) * np.log(mixtureCoefs)
        questions = np.arange(nQuestions, dtype='uint') \
            * np.ones((nPeople, 1), dtype='uint')
        H += np.sum(np.log(probs[:, questions, data - 1]), 2).T
        H -= np.reshape(np.max(H, 1), (nPeople, 1))
        H = np.exp(H)

        H /= np.reshape(np.sum(H, 1), (nPeople, 1))

        logL = loglikelihood((mixtureCoefs, probs), data, H)
        if abs(logL - oldLogL) < epsilon:
            break
        else:
            oldLogL = logL

        weights = np.sum(H, 0)
        mixtureCoefs = weights / np.sum(weights)

        for k in range(1, nAnswers + 1):
            kanswers = data == k
            for cluster in range(nClusters):
                hcluster = np.reshape(H[:, cluster], (nPeople, 1))
                probs[cluster, :, k - 1] = np.sum(kanswers * hcluster,
                        0) / weights[cluster]

    model = (mixtureCoefs, probs)
    return (model, H, logL)


def test(nb_personnes, nb_clusters, nb_questions, nb_reponses, optimise):
#     M = basicModel()
    M = generateModel(nb_questions, nb_reponses, nb_clusters)
    (D, C) = generateData(M, nb_personnes)
    if optimise:
        m, h, _ = em_opti(D, nb_clusters, 0.001)
    else:
        m, h = em_non_opti(D, nb_clusters, 1000)
    (score, mapping) = compareClusters(C, h)
    print ('Score de l\'algorithme (Distance entre les deux vecteurs) :'
           , score)
    (pm, ph) = permuteClusters(m, h, mapping)
    drawCluster(h, title='Distribution des variables cachées prédites')
    drawCluster(C, title='Distribution des vraies variables cachées')
    drawModel(M, title='Vraies distributions de réponses')
    drawModel(m, title='Distributions de réponses estimées')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nb_personnes', type=int, default=400,
                        help="nombre de personnes qui ont répondu au questionnaire")
    parser.add_argument('-c', '--nb_clusters', type=int, default=3,
                        help="nombre de clusters à détecter")
    parser.add_argument('-q', '--nb_questions', type=int, default=7,
                        help="nombre de questions")
    parser.add_argument('-a', '--nb_reponses', type=int, default=3,
                        help="nombre de réponses possibles par questions")
    parser.add_argument('-o', '--em_Optimise', type=int, default=1,
                        help="0: non optimisé, 1: optimisé",
                        choices=range(2))
    args = parser.parse_args()

    test(args.nb_personnes, args.nb_clusters, args.nb_questions, args.nb_reponses, args.em_Optimise)

    plt.waitforbuttonpress()
