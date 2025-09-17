(define (domain block-sorting)
  (:requirements :strips :typing :equality)
  (:types block table)
  (:predicates (on-table ?i - block ?t - table)
	            (handempty)
	           (robot-gripping ?i - block)
 )

  (:action pick-up
	     :parameters (?i - block ?t - table)
	     :precondition (and (on-table ?i ?t)(handempty))
	     :effect
	     (and (not (on-table ?i ?t))
		   (not (handempty))
		   (robot-gripping ?i)))

  (:action put-down
	     :parameters (?i - block ?t - table)
	     :precondition (robot-gripping ?i)
	     :effect
	     (and (not (robot-gripping ?i))
		   (handempty)
		   (on-table ?i ?t)))
)