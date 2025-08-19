(define (problem rearrange3b4l)
  (:domain rearrange-blocks)
  
  (:objects 
    block0 block1 block2 - block
    pos1 pos2 pos3 pos4 - location
  )

  (:init
    (at block0 pos1)
    (at block1 pos2)
    (at block2 pos3)
    (handempty)
    (occupied pos1)
    (occupied pos2)
    (occupied pos3)
    ;; pos4 is unoccupied
  )

  (:goal
    (and
      (at block0 pos2)
      (at block1 pos1)
      (at block2 pos3)
    )
  )
)