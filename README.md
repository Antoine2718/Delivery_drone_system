# 📦 Delivery drone system

Ce projet a pour but de résoudre un problème d'optimisation de tournée avec contraintes de capacité et d’autonomie.

Étude d'une solution initiale, puis application d'un **2-Opt intra-tournée**, suivis par un **Tabu Search** avec mouvements Relocate et Swap (inter/intra-tournée). 

Coût: distance totale + pénalités pour dépassement de capacité et d’autonomie.

Mouvements:
• Relocate: déplacer un client d’une tournée à une autre. 

• Swap: permuter deux clients entre deux tournées où au sein d’une même tournée.

• 2-Opt: amélioration locale intra-tournée après chaque mouvement accepté.
