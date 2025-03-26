(define (domain rearrange-blocks)
  (:requirements :typing :negative-preconditions)
  (:types block location)

  (:predicates
    (at ?b - block ?l - location)
    (holding ?b - block)
    (handempty)
    (occupied ?l - location)
  )

  ;; Action: pick-up a block from a location.
  (:action pick-up
    :parameters (?b - block ?l - location)
    :precondition (and 
                    (at ?b ?l)
                    (handempty))
    :effect (and 
              (holding ?b)
              (not (at ?b ?l))
              (not (handempty))
              (not (occupied ?l)))
  )

  ;; Action: put-down a block onto a location.
  ;; Precondition ensures the location is not already occupied.
  (:action put-down
    :parameters (?b - block ?l - location)
    :precondition (and 
                    (holding ?b)
                    (not (occupied ?l)))
    :effect (and 
              (at ?b ?l)
              (handempty)
              (not (holding ?b))
              (occupied ?l))
  )
)