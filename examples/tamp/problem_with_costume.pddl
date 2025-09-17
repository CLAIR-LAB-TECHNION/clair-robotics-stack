(define (problem rearrange3b4l)
  (:domain rearrange-blocks-with-costume)
  
  (:objects 
    block0 block1 block2 can - block
    pos1 pos2 pos3 pos4 pos5 - location
  )

  (:init
    (at can pos1)
    (at block0 pos4)
    (at block1 pos2)
    (at block2 pos3)
    (handempty)
    (occupied pos1)
    (occupied pos2)
    (occupied pos3)
    (occupied pos4)
    ;; pos5 is unoccupied
  )

  (:goal
    (and
      (at can pos2)
      (at block0 pos4)
      (at block1 pos5)
      (at block2 pos3)
    )
  )
)