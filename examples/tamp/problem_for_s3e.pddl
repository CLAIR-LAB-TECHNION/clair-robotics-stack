(define (problem rearrange3b4l)
  (:domain rearrange-s3e)
  
  (:objects 
    bread can - block
    pos1 pos2 pos3 - location
  )

  (:init
    (at bread pos1)
    (at can pos2)
    (handempty)
    (occupied pos1)
    (occupied pos2)
    ;; pos3 is unoccupied
  )

  (:goal
    (and
      (at bread pos1)
      (at can pos2)
    )
  )
)