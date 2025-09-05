from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class OriginalVoiceSample(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    username: str = Field(..., description="Unique username")
    clone_name: str = Field(..., description="Name for the original sample")
    original_s3_key: str = Field(...,
                                 description="MinIO S3 key path for original")
    original_minio_url: str = Field(...,
                                    description="Full MinIO URL to access the original file")
    created_at: datetime = Field(...,
                                 description="When the original was uploaded")
    note: Optional[str] = Field(
        "", description="User notes about this original sample")
    clone_ids: Optional[List[str]] = Field(
        default_factory=list, description="List of clone ObjectIds")


class VoiceClone(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    username: str = Field(..., description="Unique username")
    clone_name: str = Field(..., description="Name for the clone")
    original_id: str = Field(...,
                             description="ObjectId of the original sample")
    clone_s3_key: str = Field(..., description="MinIO S3 key path for clone")
    clone_minio_url: str = Field(...,
                                 description="Full MinIO URL to access the clone file")
    created_at: datetime = Field(..., description="When the clone was created")
    gen_text: str = Field(..., description="Text used to generate the clone")
    ref_text: str = Field(...,
                          description="Reference text from original audio")
    note: Optional[str] = Field("", description="User notes about this clone")


class UserRegistration(BaseModel):
    username: str = Field(..., min_length=1, description="Unique username")
    full_name: str = Field(..., min_length=1, description="User's full name")


class CloneCreation(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID")
    clone_name: str = Field(..., min_length=1,
                            description="Name for the clone")
    gen_text: str = Field(..., min_length=1,
                          description="Text to generate speech")
    ref_text: Optional[str] = Field(
        "", description="Reference text for training")
    note: Optional[str] = Field(
        "", description="Optional note about the clone")
    speed: Optional[float] = Field(
        1.0, ge=0.1, le=3.0, description="Speech speed")
    nfe_step: Optional[int] = Field(32, ge=1, le=100, description="NFE steps")
    cross_fade_duration: Optional[float] = Field(
        0.15, ge=0.0, le=1.0, description="Cross fade duration")


class CloneUpdate(BaseModel):
    username: str = Field(..., min_length=1, description="Username")
    old_clone_name: str = Field(..., min_length=1,
                                description="Current clone name")
    new_clone_name: str = Field(..., min_length=1,
                                description="New clone name")
    note: Optional[str] = Field("", description="Updated note about the clone")


class GradeRequest(BaseModel):
    presentation_type: str
    audience: str
    goals: List[str] = []
    custom_goals: Optional[str] = ""


class GradingFeedback(BaseModel):
    """Model for storing grading feedback data"""
    grading_id: str = Field(...,
                            description="Unique identifier for the grading session")
    user_id: Optional[str] = Field(None, description="User ID")
    original_id: Optional[str] = Field(
        None, description="Original voice sample ID")
    status: str = Field(
        ..., description="Grading status: audio_completed, script_processing, completed, failed")
    context: dict = Field(...,
                          description="Presentation context (type, audience, goals)")
    original_transcript: str = Field(...,
                                     description="Original speech transcript")
    refined_transcript: str = Field(...,
                                    description="Refined/improved transcript")
    score: float = Field(..., description="Overall audio grading score")
    overall_feedback: str = Field(...,
                                  description="Overall feedback from audio grading")
    rubric_details: List[dict] = Field(...,
                                       description="Detailed rubric scores and feedback")
    key_moments: List[dict] = Field(...,
                                    description="Key moments with timestamps")
    improvement_needed: bool = Field(...,
                                     description="Whether improvement is needed")
    script_results: Optional[dict] = Field(
        None, description="Script grading results when completed")
    error_message: Optional[str] = Field(
        None, description="Error message if grading failed")
    created_at: datetime = Field(..., description="When grading was created")
    updated_at: datetime = Field(...,
                                 description="When grading was last updated")
    completed_at: Optional[datetime] = Field(
        None, description="When grading was completed")


class AudioGradingResult(BaseModel):
    """Model for audio grading results"""
    score: float = Field(..., description="Audio grading score (1-5)")
    overall_feedback: str = Field(..., description="Overall audio feedback")
    original_transcript: str = Field(...,
                                     description="Original speech transcript")
    refined_transcript: str = Field(...,
                                    description="Refined transcript with improvements")
    rubric_details: List[dict] = Field(...,
                                       description="Detailed rubric scores")
    key_moments: List[dict] = Field(...,
                                    description="Key moments with timestamps")
    categories_evaluated: List[str] = Field(...,
                                            description="Categories that were evaluated")
    improvement_needed: bool = Field(...,
                                     description="Whether improvement is needed")


class ScriptGradingResult(BaseModel):
    """Model for script grading results"""
    original_results: dict = Field(...,
                                   description="Original script evaluation results")
    refined_results: dict = Field(...,
                                  description="Refined script evaluation results")
    comparison_analysis: dict = Field(...,
                                      description="Comparison analysis between scripts")
    summary: dict = Field(..., description="Summary of improvements")


class GradingResponse(BaseModel):
    """Model for the complete grading response"""
    grading_id: str = Field(..., description="Unique grading session ID")
    status: str = Field(..., description="Current grading status")
    script_analysis_pending: bool = Field(...,
                                          description="Whether script analysis is pending")
    poll_url: str = Field(..., description="URL to poll for status updates")
    estimated_completion: str = Field(...,
                                      description="Estimated completion time")
    audio_grading: AudioGradingResult = Field(
        ..., description="Audio grading results")
    script_grading: dict = Field(...,
                                 description="Script grading results (empty initially)")
    comprehensive_analysis: dict = Field(...,
                                         description="Comprehensive analysis results")
