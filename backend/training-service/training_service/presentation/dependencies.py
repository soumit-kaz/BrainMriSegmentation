from fastapi import Request

from training_service.application.training_session_service import TrainingSessionService


def get_training_session_service(request: Request) -> TrainingSessionService:
    return request.app.state.training_session_service
