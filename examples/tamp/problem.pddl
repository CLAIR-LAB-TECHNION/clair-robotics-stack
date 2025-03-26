(define (problem rearrange3b4l)
  (:domain rearrange-blocks)
  
  (:objects 
    red green brown - block
    pos1 pos2 pos3 pos4 - location
  )

  (:init
    (at red pos1)
    (at green pos2)
    (at brown pos3)
    (handempty)
    (occupied pos1)
    (occupied pos2)
    (occupied pos3)
    ;; pos4 is unoccupied
  )

  (:goal
    (and
      (at red pos2)
      (at green pos3)
      (at brown pos1)
    )
  )
)