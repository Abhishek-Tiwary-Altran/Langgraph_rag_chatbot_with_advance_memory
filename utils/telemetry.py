from opentelemetry import baggage, context

def set_session_context(session_id, user_id=None, actor_id=None, experiment_id=None):
    """Set session context for OpenTelemetry tracing.
    
    Args:
        session_id: Unique session identifier
        user_id: User identifier
        actor_id: Actor identifier for memory tracking
        experiment_id: Optional experiment identifier
    
    Returns:
        Context token that can be used to detach the context
    """
    ctx = baggage.set_baggage("session.id", session_id)
    
    if user_id:
        ctx = baggage.set_baggage("user.id", user_id, context=ctx)
    if actor_id:
        ctx = baggage.set_baggage("actor.id", actor_id, context=ctx)
    if experiment_id:
        ctx = baggage.set_baggage("experiment.id", experiment_id, context=ctx)
    
    return context.attach(ctx)

def get_session_context():
    """Get current session context from baggage.
    
    Returns:
        Dictionary with session context information
    """
    return {
        "session_id": baggage.get_baggage("session.id"),
        "user_id": baggage.get_baggage("user.id"),
        "actor_id": baggage.get_baggage("actor.id"),
        "experiment_id": baggage.get_baggage("experiment.id")
    }
