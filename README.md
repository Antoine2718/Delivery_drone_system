# Delivery drone system

Ce projet a pour but de résoudre un problème d'optimisation de tournée avec contraintes de capacité et d’autonomie.

Coût: distance totale + pénalités pour dépassement de capacité et d’autonomie.

Mouvements:
• Relocate: déplacer un client d’une tournée à une autre. 
• Swap: permuter deux clients entre deux tournées où au sein d’une même tournée.
• 2-Opt: amélioration locale intra-tournée après chaque mouvement accepté.
